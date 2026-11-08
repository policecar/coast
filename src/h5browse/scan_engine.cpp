// scan_engine — one scan thread, two command lanes, snapshot publication.
//
// Threading contract recap: the scan thread is the only mutator of
// model_/file_handle_/stats_working_; the GUI thread only enqueues
// commands, consumes snapshots, drains the ring, and reads the cache.
// All libhdf5 access happens in executor jobs submitted from the scan
// thread (open_and_map, read_pipeline, read_block).

#include "h5browse/scan_engine.h"

#include "h5browse/dataset_reader.h"
#include "h5browse/stats_kernel.h"

#include "hdf5/executor.h"

#include <algorithm>
#include <cassert>
#include <format>
#include <random>
#include <ranges>
#include <utility>

namespace h5b {

namespace {

// Minimal RAII guard: runs `f` on scope exit (normal or exception). Used to
// clear per-block bookkeeping on every path without a manual call at each.
template <class F>
struct scope_exit {
    F    f;
    ~scope_exit() { f(); }
};
template <class F>
scope_exit(F) -> scope_exit<F>;

std::uint64_t block_key(std::uint32_t dataset, std::uint64_t block)
{
    // Same 24-bit-dataset | 40-bit-block layout as chunk_cache::key_of;
    // the two must agree since both index the same blocks.
    assert(dataset < (1u << 24));
    assert(block < (1ull << 40));
    return (static_cast<std::uint64_t>(dataset) << 40) | block;
}

// Per-batch publication throttles: a full snapshot copy is charged on every
// forced publish, so between-batch publishes are rate-limited. Gathered here
// so the policy is one visible table rather than scattered magic numbers.
// Stats is the odd one out (33 ms, own member timer) — it copies the shared
// stats_working_ across many call sites; the rest each publish their own
// op-local `working` snapshot through throttled_pub below.
constexpr auto histogram_publish_interval = std::chrono::milliseconds{250};
constexpr auto series_publish_interval    = std::chrono::milliseconds{100};
constexpr auto pairwise_publish_interval  = std::chrono::milliseconds{100};

// Time-throttled publisher for one operation's snapshot. force=true always
// publishes (state transitions, completion); otherwise it publishes at most
// once per `interval`. Binds the destination mailbox and the op-local
// working pointer, so call sites stay `publish(force)`. Scan-thread only.
template <class T>
class throttled_pub {
public:
    throttled_pub(latest_value<std::shared_ptr<const T>>& dst,
                  const std::shared_ptr<T>& src, std::chrono::milliseconds interval)
        : dst_(&dst), src_(&src), interval_(interval)
    {
    }

    void operator()(bool force)
    {
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_ < interval_) return;
        last_ = now;
        dst_->publish(std::make_shared<T>(**src_));
    }

private:
    latest_value<std::shared_ptr<const T>>* dst_;
    const std::shared_ptr<T>*               src_;
    std::chrono::milliseconds               interval_;
    std::chrono::steady_clock::time_point   last_; // epoch by default
};

} // namespace

scan_engine::scan_engine(hdf5::executor& exec, std::function<void()> yield_hook)
    : exec_(exec), yield_hook_(std::move(yield_hook)), thread_([this] { thread_main(); })
{
}

scan_engine::~scan_engine()
{
    stop_.store(true, std::memory_order_release);
    cancel_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    thread_.join();
    // The scan thread has exited; release the session file handle (the
    // executor outlives the engine by construction: gui_manager dies
    // before main()'s executor).
    if (file_handle_ >= 0) close_file_handle(exec_, file_handle_);
}

// ----------------------------------------------------------- GUI-thread API

void scan_engine::open_file(std::filesystem::path path)
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_open{std::move(path)});
    }
    queue_cv_.notify_all();
}

void scan_engine::close_file()
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_close{});
    }
    queue_cv_.notify_all();
}

void scan_engine::compute_histograms(std::uint32_t dataset, row_filter filter)
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_histograms{dataset, std::move(filter)});
    }
    queue_cv_.notify_all();
}

void scan_engine::compute_pairwise(std::uint32_t dataset, pairwise_request request)
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_pairwise{dataset, std::move(request)});
    }
    queue_cv_.notify_all();
}

void scan_engine::compute_row_index(std::uint32_t dataset, row_filter filter)
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_row_index{dataset, std::move(filter)});
    }
    queue_cv_.notify_all();
}

void scan_engine::compute_series(std::uint32_t dataset, std::uint32_t column,
                                 row_filter filter)
{
    {
        std::scoped_lock lock(queue_mutex_);
        analysis_queue_.push_back(cmd_series{dataset, column, std::move(filter)});
    }
    queue_cv_.notify_all();
}

void scan_engine::cancel_analysis()
{
    cancel_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
}

void scan_engine::request_block(std::uint32_t dataset, std::uint64_t block)
{
    const std::uint64_t key = block_key(dataset, block);
    {
        std::scoped_lock lock(queue_mutex_);
        if (!fetch_pending_.insert(key).second) return; // already queued
        fetch_queue_.emplace_back(dataset, block);
    }
    queue_cv_.notify_all();
}

std::shared_ptr<const file_snapshot> scan_engine::consume_file()
{
    return file_pub_.get();
}
std::shared_ptr<const stats_snapshot> scan_engine::consume_stats()
{
    return stats_pub_.get();
}
std::shared_ptr<const histogram_snapshot> scan_engine::consume_histograms()
{
    return hist_pub_.get();
}
std::shared_ptr<const pairwise_snapshot> scan_engine::consume_pairwise()
{
    return pair_pub_.get();
}
std::shared_ptr<const row_index_snapshot> scan_engine::consume_row_index()
{
    return ridx_pub_.get();
}
std::shared_ptr<const series_snapshot> scan_engine::consume_series()
{
    return ser_pub_.get();
}

std::vector<log_entry> scan_engine::drain_log()
{
    std::scoped_lock       lock(log_mutex_);
    std::vector<log_entry> out(log_ring_.begin(), log_ring_.end());
    log_ring_.clear();
    return out;
}

std::shared_ptr<const row_block> scan_engine::try_get_block(std::uint32_t dataset,
                                                            std::uint64_t block)
{
    return cache_.get(dataset, block);
}

// ------------------------------------------------------------- scan thread

void scan_engine::thread_main()
{
    while (true) {
        command cmd;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] {
                return stop_.load(std::memory_order_acquire) ||
                       !analysis_queue_.empty() || !fetch_queue_.empty();
            });
            if (stop_.load(std::memory_order_acquire)) return;
            if (analysis_queue_.empty()) {
                lock.unlock();
                service_fetch_queue(); // idle table scrolling
                continue;
            }
            cmd = std::move(analysis_queue_.front());
            analysis_queue_.pop_front();
        }
        progress_.fetch_add(1, std::memory_order_relaxed);
        cancel_.store(false, std::memory_order_release); // fresh op
        // The scan thread is the top of its own call stack: an exception that
        // escapes here (e.g. bad_alloc from an oversized op) would unwind off
        // the thread and terminate the whole process. Contain it so one bad
        // operation degrades to a logged failure and the thread survives.
        try {
            std::visit(
                [this](auto&& c) {
                    using T = std::decay_t<decltype(c)>;
                    if constexpr (std::is_same_v<T, cmd_open>) {
                        run_open(std::forward<decltype(c)>(c));
                    } else if constexpr (std::is_same_v<T, cmd_close>) {
                        run_close();
                    } else if constexpr (std::is_same_v<T, cmd_histograms>) {
                        run_histograms(c);
                    } else if constexpr (std::is_same_v<T, cmd_pairwise>) {
                        run_pairwise(c);
                    } else if constexpr (std::is_same_v<T, cmd_row_index>) {
                        run_row_index(std::forward<decltype(c)>(c));
                    } else if constexpr (std::is_same_v<T, cmd_series>) {
                        run_series(c);
                    }
                },
                std::move(cmd));
        } catch (const std::exception& e) {
            log(log_severity::error,
                std::format("scan operation failed: {}", e.what()));
        } catch (...) {
            log(log_severity::error,
                "scan operation failed: unknown exception");
        }
    }
}

bool scan_engine::should_yield()
{
    progress_.fetch_add(1, std::memory_order_relaxed);
    if (yield_hook_) yield_hook_();
    if (stop_.load(std::memory_order_acquire)) return true;
    if (cancel_.load(std::memory_order_acquire)) return true;
    std::scoped_lock lock(queue_mutex_);
    for (const command& c : analysis_queue_) {
        if (std::holds_alternative<cmd_open>(c) ||
            std::holds_alternative<cmd_close>(c)) {
            return true; // a new session supersedes the running analysis
        }
    }
    return false;
}

const dataset_desc* scan_engine::require_open_dataset(std::uint32_t idx,
                                                      std::string_view op)
{
    if (!model_ || idx >= model_->datasets.size()) {
        log(log_severity::warning,
            std::format("{} request without a matching open dataset — ignored",
                        op));
        return nullptr;
    }
    return &model_->datasets[idx];
}

void scan_engine::service_fetch_queue()
{
    while (true) {
        // Shutdown must not wait for a large queued backlog to drain: the
        // destructor sets stop_ then joins, so bail here to keep close
        // responsive. cancel_ is intentionally NOT checked — fetches are meant
        // to survive an analysis-cancel.
        if (stop_.load(std::memory_order_acquire)) return;
        std::pair<std::uint32_t, std::uint64_t> req;
        {
            std::scoped_lock lock(queue_mutex_);
            if (fetch_queue_.empty()) return;
            req = fetch_queue_.front();
            fetch_queue_.pop_front();
        }
        progress_.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t key = block_key(req.first, req.second);
        // The dedupe entry is removed AFTER servicing so re-requests during
        // the read collapse into this one. RAII so it clears on every exit
        // path — including an exception (e.g. OOM in cache_.put/log): a leaked
        // entry would silently drop all future requests for this block.
        const scope_exit release{[&] {
            std::scoped_lock lock(queue_mutex_);
            fetch_pending_.erase(key);
        }};

        if (!model_ || req.first >= model_->datasets.size()) {
            continue; // stale request from a closed/previous session
        }
        if (cache_.get(req.first, req.second) != nullptr) {
            continue;
        }
        if (failed_blocks_.contains(key)) {
            continue; // already reported; GUI shows the log entry
        }

        const dataset_desc& d   = model_->datasets[req.first];
        const std::uint64_t rpb = default_block_rows(d);
        auto blk = read_block(exec_, file_handle_, d, req.first, req.second, rpb);
        if (blk) {
            cache_.put(std::move(*blk));
        } else {
            failed_blocks_.insert(key);
            log(log_severity::error,
                std::format("table fetch failed: {} block {}: {}",
                            d.display_name, req.second, blk.error().message));
        }
    }
}

void scan_engine::publish_file()
{
    auto snap = std::make_shared<file_snapshot>();
    if (model_) {
        snap->open  = true;
        snap->model = *model_;
    }
    file_pub_.publish(std::move(snap));
}

void scan_engine::publish_stats(bool force)
{
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_stats_publish_ < std::chrono::milliseconds{33}) {
        return;
    }
    last_stats_publish_ = now;
    stats_pub_.publish(std::make_shared<stats_snapshot>(stats_working_));
}

void scan_engine::log(log_severity sev, std::string message)
{
    std::scoped_lock lock(log_mutex_);
    if (log_ring_.size() >= log_ring_capacity) log_ring_.pop_front();
    log_ring_.push_back(
        log_entry{sev, std::chrono::system_clock::now(), std::move(message)});
}

// -------------------------------------------------------------- operations

void scan_engine::run_open(cmd_open&& cmd)
{
    run_close(); // one file at a time; also resets all working state

    log(log_severity::info, std::format("opening {}", cmd.path.string()));
    auto mapped = open_and_map(exec_, cmd.path);
    if (!mapped) {
        const auto& e = mapped.error();
        log(log_severity::error,
            std::format("open rejected [{}] {}: {}", error_category_name(e.category),
                        e.object_path, e.message));
        auto snap           = std::make_shared<file_snapshot>();
        snap->error_message = e.message;
        file_pub_.publish(std::move(snap));
        return;
    }

    auto handle = open_file_readonly(exec_, cmd.path);
    if (!handle) {
        log(log_severity::error,
            std::format("session handle failed: {}", handle.error().message));
        auto snap           = std::make_shared<file_snapshot>();
        snap->error_message = handle.error().message;
        file_pub_.publish(std::move(snap));
        return;
    }

    model_       = std::move(*mapped);
    file_handle_ = *handle;
    stats_working_.per_dataset.assign(model_->datasets.size(), {});
    stats_working_.scan_complete = false;

    log(log_severity::info,
        std::format("opened {} — family {}, {} datasets, {} MB", cmd.path.string(),
                    file_family_name(model_->family), model_->datasets.size(),
                    model_->file_bytes >> 20));
    publish_file();
    publish_stats(true);

    run_stats_scan(); // Tier-2 auto scan (D4)
}

void scan_engine::run_close()
{
    if (!model_) return;
    if (file_handle_ >= 0) {
        close_file_handle(exec_, file_handle_);
        file_handle_ = -1;
    }
    model_.reset();
    stats_working_ = {};
    cache_.clear();
    failed_blocks_.clear();
    // Drop table-read work queued against the file being closed, so it can't
    // be serviced against the next file (whose datasets share the same
    // indices). Both structures are guarded by queue_mutex_.
    {
        std::scoped_lock lock(queue_mutex_);
        fetch_queue_.clear();
        fetch_pending_.clear();
    }
    publish_file();
    publish_stats(true);
    log(log_severity::info, "file closed");
}

void scan_engine::run_stats_scan()
{
    for (std::uint32_t i = 0; i < model_->datasets.size(); ++i) {
        const dataset_desc&  d     = model_->datasets[i];
        dataset_stats_entry& entry = stats_working_.per_dataset[i];
        entry.state                = op_state::running;

        read_pipeline pipe(exec_, file_handle_, d, 0, d.rows);
        bool          aborted = false;
        while (true) {
            if (should_yield()) {
                entry.state = op_state::cancelled;
                log(log_severity::warning,
                    std::format("stats scan cancelled at {} ({} of {} rows)",
                                d.display_name, entry.rows_scanned, d.rows));
                aborted = true;
                break;
            }
            auto b = pipe.next();
            if (!b) {
                entry.state         = op_state::failed;
                entry.error_message = b.error().message;
                log(log_severity::error,
                    std::format("stats scan failed: {}: {}", d.display_name,
                                b.error().message));
                break; // containment: next dataset continues
            }
            if (!b->has_value()) {
                entry.state = op_state::done;
                break;
            }
            const auto& batch = **b;
            dispatch(d.dtype, [&](auto tag) {
                using T = typename decltype(tag)::type;
                accumulate_stats<T>(
                    entry.stats,
                    std::span<const T>(
                        reinterpret_cast<const T*>(batch.data.data()),
                        batch.n_rows * d.row_width));
            });
            entry.rows_scanned += batch.n_rows;
            publish_stats(false); // throttled progress
            service_fetch_queue(); // D9a: table stays live mid-scan
        }
        publish_stats(true); // state transition
        if (aborted) return; // remaining datasets stay pending
    }
    stats_working_.scan_complete = true;
    publish_stats(true);

    std::uint64_t failed = 0;
    for (const auto& e : stats_working_.per_dataset) {
        if (e.state == op_state::failed) ++failed;
    }
    if (failed > 0) {
        log(log_severity::warning,
            std::format("stats scan complete: {} of {} datasets unreadable",
                        failed, stats_working_.per_dataset.size()));
    } else {
        log(log_severity::info, "stats scan complete");
    }
}

void scan_engine::run_histograms(const cmd_histograms& cmd)
{
    const dataset_desc* dp = require_open_dataset(cmd.dataset, "histogram");
    if (!dp) return;
    const dataset_desc& d = *dp;

    auto working         = std::make_shared<histogram_snapshot>();
    working->dataset     = cmd.dataset;
    working->filter_desc = cmd.filter.describe();
    working->state       = op_state::running;

    // Publication is time-throttled: a max-width column set is ~64 MB per
    // copy, so per-batch publication would dominate the pass.
    throttled_pub<histogram_snapshot> publish(hist_pub_, working,
                                              histogram_publish_interval);

    if (d.row_width > max_histogram_columns) {
        working->state         = op_state::failed;
        working->error_message = std::format(
            "dataset is {} columns wide; per-column histograms are capped "
            "at {} columns",
            d.row_width, max_histogram_columns);
        log(log_severity::error, std::format("histograms {}: {}",
                                             d.display_name,
                                             working->error_message));
        publish(true);
        return;
    }

    working->columns.assign(d.row_width, base_histogram(per_column_base_bins));
    log(log_severity::info,
        std::format("histograms: {} ({} columns, filter: {})", d.display_name,
                    d.row_width, working->filter_desc));
    publish(true);

    read_pipeline pipe(exec_, file_handle_, d, 0, d.rows);
    while (true) {
        if (should_yield()) {
            working->state = op_state::cancelled;
            log(log_severity::warning,
                std::format("histograms cancelled at {} of {} rows",
                            working->rows_scanned, d.rows));
            break;
        }
        auto b = pipe.next();
        if (!b) {
            working->state         = op_state::failed;
            working->error_message = b.error().message;
            log(log_severity::error,
                std::format("histograms failed: {}: {}", d.display_name,
                            b.error().message));
            break;
        }
        if (!b->has_value()) {
            working->state = op_state::done;
            break;
        }
        const auto& batch = **b;
        working->rows_matched += dispatch(d.dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            return insert_histograms<T>(
                working->columns,
                std::span<const T>(
                    reinterpret_cast<const T*>(batch.data.data()),
                    batch.n_rows * d.row_width),
                d.row_width, cmd.filter, batch.first_row);
        });
        working->rows_scanned += batch.n_rows;
        publish(false);
        service_fetch_queue(); // D9a
    }
    publish(true);
    if (working->state == op_state::done) {
        log(log_severity::info,
            std::format("histograms done: {} — {} of {} rows matched",
                        d.display_name, working->rows_matched,
                        working->rows_scanned));
    }
}

void scan_engine::run_row_index(cmd_row_index&& cmd)
{
    const dataset_desc* dp = require_open_dataset(cmd.dataset, "row-index");
    if (!dp) return;
    const dataset_desc& d = *dp;

    auto working         = std::make_shared<row_index_snapshot>();
    working->dataset     = cmd.dataset;
    working->filter_desc = cmd.filter.describe();
    working->state       = op_state::running;
    ridx_pub_.publish(std::make_shared<row_index_snapshot>(*working));

    auto last_publish = std::chrono::steady_clock::time_point{};
    // Progress copies exclude the (potentially huge) match list, so this
    // stays a bespoke publisher rather than throttled_pub (which copies the
    // whole working snapshot). Interval matches series/pairwise (100 ms).
    auto publish_progress = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_publish < series_publish_interval) return;
        last_publish   = now;
        auto snap      = std::make_shared<row_index_snapshot>();
        snap->dataset  = working->dataset;
        snap->state    = working->state;
        snap->filter_desc  = working->filter_desc;
        snap->rows_scanned = working->rows_scanned;
        snap->rows_matched = working->rows_matched;
        ridx_pub_.publish(std::move(snap));
    };

    read_pipeline pipe(exec_, file_handle_, d, 0, d.rows);
    while (true) {
        if (should_yield()) {
            working->state = op_state::cancelled;
            working->rows.clear();
            log(log_severity::warning,
                std::format("row filter cancelled at {} of {} rows",
                            working->rows_scanned, d.rows));
            break;
        }
        auto b = pipe.next();
        if (!b) {
            working->state         = op_state::failed;
            working->error_message = b.error().message;
            working->rows.clear();
            log(log_severity::error,
                std::format("row filter failed: {}: {}", d.display_name,
                            b.error().message));
            break;
        }
        if (!b->has_value()) {
            working->state = op_state::done;
            break;
        }
        const auto& batch = **b;
        dispatch(d.dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            collect_matching_rows<T>(
                working->rows,
                std::span<const T>(
                    reinterpret_cast<const T*>(batch.data.data()),
                    batch.n_rows * d.row_width),
                d.row_width, cmd.filter, batch.first_row);
        });
        working->rows_scanned += batch.n_rows;
        working->rows_matched = working->rows.size();
        publish_progress();
        service_fetch_queue(); // D9a
    }
    working->rows_matched = working->rows.size();
    ridx_pub_.publish(std::move(working)); // final: includes the match list
}

void scan_engine::run_series(const cmd_series& cmd)
{
    const dataset_desc* dp = require_open_dataset(cmd.dataset, "series");
    if (!dp) return;
    const dataset_desc& d = *dp;

    auto working         = std::make_shared<series_snapshot>();
    working->dataset     = cmd.dataset;
    working->column      = cmd.column;
    working->filter_desc = cmd.filter.describe();
    working->state       = op_state::running;

    throttled_pub<series_snapshot> publish(ser_pub_, working,
                                           series_publish_interval);

    if (cmd.column >= d.row_width) {
        working->error_message = std::format(
            "column {} out of range (width {})", cmd.column, d.row_width);
        working->state = op_state::failed;
        log(log_severity::error,
            std::format("series {}: {}", d.display_name, working->error_message));
        publish(true);
        return;
    }

    // Scan span: the filter's row range bounds the read itself.
    const std::uint64_t begin = cmd.filter.row_begin.value_or(0);
    const std::uint64_t end =
        std::min<std::uint64_t>(cmd.filter.row_end.value_or(d.rows), d.rows);
    if (begin >= end) {
        working->state     = op_state::done; // empty span: legal, no data
        working->row_begin = begin;
        working->row_end   = begin;
        publish(true);
        return;
    }
    const std::uint64_t span = end - begin;
    const auto          n_buckets = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(series_max_buckets, span));
    working->row_begin = begin;
    working->row_end   = end;
    working->buckets.assign(n_buckets, {});
    publish(true);

    read_pipeline pipe(exec_, file_handle_, d, begin, span);
    while (true) {
        if (should_yield()) {
            working->state = op_state::cancelled;
            log(log_severity::warning,
                std::format("series cancelled at {} of {} rows",
                            working->rows_scanned, span));
            break;
        }
        auto b = pipe.next();
        if (!b) {
            working->state         = op_state::failed;
            working->error_message = b.error().message;
            log(log_severity::error,
                std::format("series failed: {}: {}", d.display_name,
                            b.error().message));
            break;
        }
        if (!b->has_value()) {
            working->state = op_state::done;
            break;
        }
        const auto& batch = **b;
        dispatch(d.dtype, [&](auto tag) {
            using T      = typename decltype(tag)::type;
            const T* ptr = reinterpret_cast<const T*>(batch.data.data());
            for (std::uint64_t r = 0; r < batch.n_rows; ++r) {
                const std::uint64_t row = batch.first_row + r;
                const std::span<const T> rowspan(ptr + r * d.row_width,
                                                 d.row_width);
                if (!row_matches(cmd.filter, rowspan)) continue;
                const double v = static_cast<double>(rowspan[cmd.column]);
                if (!std::isfinite(v)) continue; // NaN/Inf: stats cover them
                const auto bidx = static_cast<std::size_t>(
                    (row - begin) * n_buckets / span);
                series_bucket& bk = working->buckets[bidx];
                if (bk.count == 0) {
                    bk.min = bk.max = v;
                } else {
                    bk.min = std::min(bk.min, v);
                    bk.max = std::max(bk.max, v);
                }
                bk.sum += v;
                ++bk.count;
                ++working->rows_matched;
            }
        });
        working->rows_scanned += batch.n_rows;
        publish(false);
        service_fetch_queue(); // D9a
    }
    publish(true);
    if (working->state == op_state::done) {
        log(log_severity::info,
            std::format("series done: {} col {} — {} of {} rows",
                        d.display_name, cmd.column, working->rows_matched,
                        span));
    }
}

result<std::vector<double>>
scan_engine::gather_rows_f64(const dataset_desc& d, std::uint32_t dataset_index,
                             std::span<const std::uint64_t> rows)
{
    const std::uint64_t rpb = default_block_rows(d);
    std::vector<double> out(rows.size() * d.row_width);

    std::shared_ptr<const row_block> blk;
    std::size_t                      oi = 0;
    for (const std::uint64_t idx : rows) {
        const std::uint64_t block = idx / rpb;
        if (!blk || blk->block != block) {
            blk = cache_.get(dataset_index, block);
            if (!blk) {
                auto r = read_block(exec_, file_handle_, d, dataset_index,
                                    block, rpb);
                if (!r) return std::unexpected(r.error());
                cache_.put(*r);
                blk = std::move(*r);
            }
        }
        const std::uint64_t off = idx - blk->first_row;
        dispatch(d.dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            const T* src = reinterpret_cast<const T*>(blk->bytes.data()) +
                           off * d.row_width;
            double* dst = out.data() + oi * d.row_width;
            for (std::uint64_t c = 0; c < d.row_width; ++c) {
                dst[c] = static_cast<double>(src[c]);
            }
        });
        ++oi;
    }
    return out;
}

void scan_engine::run_pairwise(const cmd_pairwise& cmd)
{
    const dataset_desc* dp = require_open_dataset(cmd.dataset, "pairwise");
    if (!dp) return;
    const dataset_desc& d = *dp;

    auto working       = std::make_shared<pairwise_snapshot>();
    working->dataset   = cmd.dataset;
    working->request   = cmd.request;
    working->state     = op_state::running;
    working->histogram = make_metric_histogram(cmd.request.kind);

    throttled_pub<pairwise_snapshot> publish(pair_pub_, working,
                                             pairwise_publish_interval);
    auto finish = [&](op_state st, std::string err = {}) {
        working->state         = st;
        working->error_message = std::move(err);
        publish(true);
    };
    publish(true);

    // Refuse absurdly wide rows before gathering: the dense f64 tile and the
    // O(pairs * width) compare would otherwise attempt a huge allocation /
    // runtime. Mirrors the histogram path's width guard.
    if (d.row_width > max_pairwise_columns) {
        const std::string msg = std::format(
            "dataset is {} columns wide; pairwise comparison is capped at "
            "{} columns",
            d.row_width, max_pairwise_columns);
        log(log_severity::error,
            std::format("pairwise {}: {}", d.display_name, msg));
        finish(op_state::failed, msg);
        return;
    }

    // 1. Filtered population (explicit indices only when a filter is set).
    const bool                 unfiltered = cmd.request.filter.empty();
    std::vector<std::uint64_t> pop;
    if (!unfiltered) {
        read_pipeline pipe(exec_, file_handle_, d, 0, d.rows);
        while (true) {
            if (should_yield()) {
                log(log_severity::warning, "pairwise cancelled during filter pass");
                finish(op_state::cancelled);
                return;
            }
            auto b = pipe.next();
            if (!b) {
                log(log_severity::error,
                    std::format("pairwise filter pass failed: {}: {}",
                                d.display_name, b.error().message));
                finish(op_state::failed, b.error().message);
                return;
            }
            if (!b->has_value()) break;
            const auto& batch = **b;
            dispatch(d.dtype, [&](auto tag) {
                using T = typename decltype(tag)::type;
                collect_matching_rows<T>(
                    pop,
                    std::span<const T>(
                        reinterpret_cast<const T*>(batch.data.data()),
                        batch.n_rows * d.row_width),
                    d.row_width, cmd.request.filter, batch.first_row);
            });
            service_fetch_queue();
        }
    }
    const std::uint64_t m_pop   = unfiltered ? d.rows : pop.size();
    working->rows_in_filter     = m_pop;

    // 2. Selection: everything (full run) or k sampled without replacement.
    // std::sample preserves population order, so selections stay sorted
    // (gather locality). Full unfiltered runs stay implicit — tiles
    // generate their iota indices on the fly, nothing rows-sized is
    // materialized.
    std::vector<std::uint64_t> sel;
    bool                       implicit_sel = false;
    std::uint64_t              m            = 0;
    if (cmd.request.full_run) {
        m = m_pop;
        if (unfiltered) {
            implicit_sel = true;
        } else {
            sel = std::move(pop);
        }
    } else {
        m = std::min<std::uint64_t>(cmd.request.k, m_pop);
        sel.resize(m);
        std::mt19937_64 rng(cmd.request.seed);
        if (unfiltered) {
            auto all = std::views::iota(std::uint64_t{0}, m_pop);
            std::sample(all.begin(), all.end(), sel.begin(), m, rng);
        } else {
            std::sample(pop.begin(), pop.end(), sel.begin(), m, rng);
        }
    }
    if (m < 2) {
        log(log_severity::warning,
            std::format("pairwise: fewer than 2 rows selected ({} in filter)",
                        m_pop));
        finish(op_state::done);
        return;
    }
    working->pairs_total = m * (m - 1) / 2;
    publish(true);
    log(log_severity::info,
        std::format("pairwise: {} — {} rows, {} pairs ({}, {})",
                    d.display_name, m, working->pairs_total,
                    coast::metric_kind_name(cmd.request.kind),
                    cmd.request.full_run ? "full run" : "sampled"));

    // 3. Tile-pair iteration (sampled default = one tile).
    const std::uint64_t n_tiles =
        (m + pairwise_tile_rows - 1) / pairwise_tile_rows;
    auto tile_indices = [&](std::uint64_t t) {
        const std::uint64_t lo = t * pairwise_tile_rows;
        const std::uint64_t hi = std::min(m, lo + pairwise_tile_rows);
        std::vector<std::uint64_t> idx;
        idx.reserve(hi - lo);
        for (std::uint64_t i = lo; i < hi; ++i) {
            idx.push_back(implicit_sel ? i : sel[i]);
        }
        return idx;
    };

    for (std::uint64_t ta = 0; ta < n_tiles; ++ta) {
        const auto a = gather_rows_f64(d, cmd.dataset, tile_indices(ta));
        if (!a) {
            log(log_severity::error, std::format("pairwise gather failed: {}",
                                                 a.error().message));
            finish(op_state::failed, a.error().message);
            return;
        }
        const std::uint64_t a_rows = std::min(m - ta * pairwise_tile_rows,
                                              pairwise_tile_rows);
        for (std::uint64_t tb = ta; tb < n_tiles; ++tb) {
            if (should_yield()) {
                log(log_severity::warning,
                    std::format("pairwise cancelled at {} of {} pairs",
                                working->pairs_done, working->pairs_total));
                finish(op_state::cancelled);
                return;
            }
            std::vector<double> b_store;
            const double*       b_data = a->data();
            std::uint64_t       b_rows = a_rows;
            if (tb != ta) {
                auto b = gather_rows_f64(d, cmd.dataset, tile_indices(tb));
                if (!b) {
                    log(log_severity::error,
                        std::format("pairwise gather failed: {}",
                                    b.error().message));
                    finish(op_state::failed, b.error().message);
                    return;
                }
                b_store = std::move(*b);
                b_data  = b_store.data();
                b_rows  = std::min(m - tb * pairwise_tile_rows,
                                   pairwise_tile_rows);
            }
            working->pairs_done += pairwise_tile(
                std::span<const double>(a->data(), a_rows * d.row_width),
                a_rows,
                std::span<const double>(b_data, b_rows * d.row_width), b_rows,
                d.row_width, tb == ta, cmd.request.kind, cmd.request.params,
                working->histogram);
            publish(false);
            service_fetch_queue(); // D9a
        }
    }
    finish(op_state::done);
    log(log_severity::info,
        std::format("pairwise done: {} pairs, {} in histogram",
                    working->pairs_done, working->histogram.total()));
}

} // namespace h5b
