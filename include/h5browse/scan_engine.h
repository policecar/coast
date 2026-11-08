#ifndef COAST_SCAN_ENGINE_H
#define COAST_SCAN_ENGINE_H

// h5browse scan engine (IDD §6/§8): owns the one scan thread, the two
// command lanes, the chunk cache, the log ring, and all accumulators.
//
//   GUI thread  ──commands──▶  scan thread  ──chunk read jobs──▶  executor
//   GUI thread  ◀─snapshots──  (demand_published, latest-state-wins)
//
// Two-lane servicing (D9a): analysis commands (open/close/histograms/
// pairwise) are strictly serialized on the scan thread and cancellable
// per batch; block-fetch requests are serviced BETWEEN batches of any
// running analysis (fetches are ~ms), so the table view stays live during
// long scans. cancel_analysis never cancels fetches.
//
// Snapshot transport (deviation from IDD D3, discovered by test):
// demand_published's contract needs a producer that keeps re-offering
// (the sim thread calls produce_if_wanted every step); this engine is
// event-driven — after an operation's last publish nothing offers again,
// so a skipped produce_if_wanted would leave the GUI on a stale snapshot
// forever. Snapshots therefore use latest_value: an unconditional
// mutex-guarded latest-state mailbox (publisher pays one snapshot copy
// per publish; publication is time-throttled during scans).
// This header stays ImGui-free.

#include "h5browse/base_histogram.h"
#include "h5browse/chunk_cache.h"
#include "h5browse/file_model.h"
#include "h5browse/pairwise.h"
#include "h5browse/row_filter.h"
#include "h5browse/scan_stats.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <variant>
#include <vector>

namespace hdf5 {
class executor;
}

namespace h5b {

// ------------------------------------------------------------- snapshots

enum class op_state : std::uint8_t { pending, running, done, failed, cancelled };

struct file_snapshot {
    bool        open = false;
    file_model  model;         // valid when open
    std::string error_message; // set when the last open was rejected
};

struct dataset_stats_entry {
    op_state      state = op_state::pending;
    scan_stats    stats;            // partial while running (labeled by state)
    std::uint64_t rows_scanned = 0;
    std::string   error_message;    // failed datasets (D8 containment)
};

struct stats_snapshot {
    std::vector<dataset_stats_entry> per_dataset; // parallel to model.datasets
    bool                             scan_complete = false;
};

inline constexpr std::uint32_t per_column_base_bins = 512; // IDD-pinned

struct histogram_snapshot {
    std::uint32_t               dataset = 0;
    op_state                    state   = op_state::pending;
    std::vector<base_histogram> columns; // one per column, 512 base bins
    std::string                 filter_desc;
    std::uint64_t               rows_scanned = 0;
    std::uint64_t               rows_matched = 0;
    std::string                 error_message;
};

// Value-over-rows graph, decimated on the scan thread: one bucket per
// ~span/n rows carrying min/max and the running sum (mean = sum/count) of
// the matching rows' column values (empty buckets have count == 0). Sized
// for direct plotting.
inline constexpr std::uint32_t series_max_buckets = 2048;

struct series_bucket {
    double        min   = 0.0;
    double        max   = 0.0;
    double        sum   = 0.0;
    std::uint64_t count = 0;
};

struct series_snapshot {
    std::uint32_t dataset = 0;
    std::uint32_t column  = 0;
    op_state      state   = op_state::pending;
    std::string   filter_desc;
    std::uint64_t row_begin = 0, row_end = 0; // scanned span
    std::uint64_t rows_scanned = 0;
    std::uint64_t rows_matched = 0;
    std::vector<series_bucket> buckets;
    std::string   error_message;
};

// Filtered table view: absolute indices of matching rows (<= 8 B/row).
// `rows` is only populated in the final (done) publication; progress
// publications carry rows_scanned/rows_matched only (a large match list
// must not be copied per batch).
struct row_index_snapshot {
    std::uint32_t dataset = 0;
    op_state      state   = op_state::pending;
    std::string   filter_desc;
    std::uint64_t rows_scanned = 0;
    std::uint64_t rows_matched = 0;
    std::vector<std::uint64_t> rows;
    std::string   error_message;
};

struct pairwise_snapshot {
    std::uint32_t    dataset = 0;
    op_state         state   = op_state::pending;
    pairwise_request request;      // echo of what is being computed
    base_histogram   histogram{pairwise_base_bins};
    std::uint64_t    pairs_done  = 0;
    std::uint64_t    pairs_total = 0;
    std::uint64_t    rows_in_filter = 0; // filtered population size
    std::string      error_message;
};

// --------------------------------------------------------------- log ring

enum class log_severity : std::uint8_t { info, warning, error };

struct log_entry {
    log_severity                          severity = log_severity::info;
    std::chrono::system_clock::time_point time;
    std::string                           message;
};

// Latest-state-wins mailbox: one producer (scan thread), one consumer
// (GUI). get() is cheap (shared_ptr copy under a short lock).
template <class T>
class latest_value {
public:
    void publish(T v)
    {
        std::scoped_lock lock(mutex_);
        value_ = std::move(v);
    }
    [[nodiscard]] T get() const
    {
        std::scoped_lock lock(mutex_);
        return value_;
    }

private:
    mutable std::mutex mutex_;
    T                  value_{};
};

// ------------------------------------------------------------ scan_engine

class scan_engine {
public:
    static constexpr std::size_t log_ring_capacity = 4096;
    static constexpr std::size_t max_histogram_columns = 16384; // width cap
    // Pairwise gathers a dense tile (<=2 * pairwise_tile_rows) as f64; peak
    // memory and per-pair compute scale with width, so cap it. Distinct from
    // the histogram cap (which bounds per-column state), same value for now.
    static constexpr std::size_t max_pairwise_columns = 16384;

    // `yield_hook` (test seam; empty in production) runs on the scan thread at
    // the top of every yield point of a running analysis — once per scanned
    // batch — before the cancel/supersede checks. A test can hold the scan
    // there to act while an analysis is provably in progress. A hook that
    // blocks must be released before the engine is destroyed.
    explicit scan_engine(hdf5::executor& exec, std::function<void()> yield_hook = {});
    ~scan_engine(); // cancels, wakes, joins the scan thread

    // Progress counter of the scan thread (read-only observation point): it
    // advances on every dispatched command, every yield point of a running
    // analysis and every serviced block fetch. Tests use it as a heartbeat.
    [[nodiscard]] std::uint64_t progress() const noexcept {
        return progress_.load(std::memory_order_relaxed);
    }

    scan_engine(const scan_engine&)            = delete;
    scan_engine& operator=(const scan_engine&) = delete;

    // -- commands (GUI thread; non-blocking enqueue) --
    void open_file(std::filesystem::path path);
    void close_file();
    void compute_histograms(std::uint32_t dataset, row_filter filter);
    void compute_pairwise(std::uint32_t dataset, pairwise_request request);
    void compute_row_index(std::uint32_t dataset, row_filter filter);
    void compute_series(std::uint32_t dataset, std::uint32_t column,
                        row_filter filter);
    void cancel_analysis(); // current analysis only; fetches unaffected
    void request_block(std::uint32_t dataset, std::uint64_t block); // deduplicated

    // -- per-frame consumption (GUI thread) --
    // nullptr until first publication; latest-state-wins thereafter.
    std::shared_ptr<const file_snapshot>      consume_file();
    std::shared_ptr<const stats_snapshot>     consume_stats();
    std::shared_ptr<const histogram_snapshot> consume_histograms();
    std::shared_ptr<const pairwise_snapshot>  consume_pairwise();
    std::shared_ptr<const row_index_snapshot> consume_row_index();
    std::shared_ptr<const series_snapshot>    consume_series();

    std::vector<log_entry>           drain_log();
    std::shared_ptr<const row_block> try_get_block(std::uint32_t dataset,
                                                   std::uint64_t block);

private:
    struct cmd_open {
        std::filesystem::path path;
    };
    struct cmd_close {};
    struct cmd_histograms {
        std::uint32_t dataset;
        row_filter    filter;
    };
    struct cmd_pairwise {
        std::uint32_t    dataset;
        pairwise_request request;
    };
    struct cmd_row_index {
        std::uint32_t dataset;
        row_filter    filter;
    };
    struct cmd_series {
        std::uint32_t dataset;
        std::uint32_t column;
        row_filter    filter;
    };
    using command = std::variant<cmd_open, cmd_close, cmd_histograms,
                                 cmd_pairwise, cmd_row_index, cmd_series>;

    void thread_main();
    void run_open(cmd_open&& cmd);
    void run_close();
    void run_stats_scan();
    void run_histograms(const cmd_histograms& cmd);
    void run_pairwise(const cmd_pairwise& cmd);
    void run_row_index(cmd_row_index&& cmd);
    void run_series(const cmd_series& cmd);

    // True when the running analysis should stop: cancel flag, engine
    // shutdown, or a queued open/close that supersedes it.
    [[nodiscard]] bool should_yield();

    // The dataset_desc for `idx`, or nullptr (with a warning naming `op`)
    // when no file is open or the index is out of range. Centralizes the
    // "request without a matching open dataset" guard shared by every
    // analysis op.
    [[nodiscard]] const dataset_desc* require_open_dataset(std::uint32_t idx,
                                                           std::string_view op);

    // Gathers the given (sorted) absolute rows into a dense f64 matrix
    // through the chunk cache — the browser's only dtype conversion (D10).
    // f64 (not f32) so wide integer sources keep full precision.
    result<std::vector<double>> gather_rows_f64(const dataset_desc& d,
                                                std::uint32_t dataset_index,
                                                std::span<const std::uint64_t> rows);

    void service_fetch_queue(); // between batches and when idle
    void publish_file();
    // force=true at state transitions and completion; otherwise throttled
    // to ~30 Hz so per-batch publication stays cheap on wide files.
    void publish_stats(bool force);
    void log(log_severity sev, std::string message);

    hdf5::executor& exec_;

    // -- queues (GUI -> scan thread) --
    std::mutex              queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<command>     analysis_queue_;
    std::deque<std::pair<std::uint32_t, std::uint64_t>> fetch_queue_;
    std::unordered_set<std::uint64_t> fetch_pending_; // dedupe keys
    std::atomic<bool>                 cancel_{false};
    std::atomic<bool>                 stop_{false};

    // -- scan-thread state (no lock needed; single owner) --
    std::optional<file_model> model_;
    std::int64_t              file_handle_ = -1;
    stats_snapshot            stats_working_;
    std::unordered_set<std::uint64_t> failed_blocks_; // logged once each
    std::chrono::steady_clock::time_point last_stats_publish_;

    // -- shared state --
    chunk_cache            cache_;
    std::mutex             log_mutex_;
    std::deque<log_entry>  log_ring_;

    // -- snapshots (scan thread produces, GUI consumes) --
    latest_value<std::shared_ptr<const file_snapshot>>      file_pub_;
    latest_value<std::shared_ptr<const stats_snapshot>>     stats_pub_;
    latest_value<std::shared_ptr<const histogram_snapshot>> hist_pub_;
    latest_value<std::shared_ptr<const pairwise_snapshot>>  pair_pub_;
    latest_value<std::shared_ptr<const row_index_snapshot>> ridx_pub_;
    latest_value<std::shared_ptr<const series_snapshot>>    ser_pub_;

    std::function<void()>        yield_hook_;      // test seam, see ctor
    std::atomic<std::uint64_t>   progress_{0};     // see progress()
    std::thread thread_; // last member: starts in ctor after all state
};

} // namespace h5b

#endif // COAST_SCAN_ENGINE_H
