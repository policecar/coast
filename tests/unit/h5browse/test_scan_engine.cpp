// h5browse scan_engine end-to-end (headless): open -> auto stats scan ->
// snapshot convergence; poisoned-dataset containment (D8); fetch lane
// serving blocks during an active scan and after it (D9a); cancellation
// (op_state::cancelled reached, whole sequence aborts) and session
// supersede; close_file teardown + reopen; open-failure reporting. All
// through the real executor.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/scan_engine.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include "await_progress.h"

#include <hdf5.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Wait for pred() on the engine's own progress (scan_engine::progress()), not
// on a deadline: a loaded machine only makes the wait longer (CLAUDE.md
// "Testing"). Fails only if the scan thread stops making progress.
bool eventually(h5b::scan_engine& engine, const std::function<bool()>& pred)
{
    // The heartbeat is the engine's work counter, and its work is finite: scan
    // batches are >= 4 MB and pairwise tiles 4096 rows, so the inputs here need
    // at most a few hundred yield points. Far more than any of them could need
    // means an operation is looping — that ends the wait (and fails), instead of
    // a heartbeat that ticks forever.
    constexpr std::uint64_t k_max_work = 100'000;
    const std::uint64_t start = engine.progress();
    bool held = false;
    (void)coast_test::await_progress(
        [&] {
            held = pred();
            return held || engine.progress() - start > k_max_work;
        },
        [&] { return engine.progress(); });
    return held;
}

// Holds the scan thread at its yield points (scan_engine's yield_hook seam):
// yield point n (1-based) passes only once allow(n) was called, so the test
// acts while an analysis is provably in progress. Releases everything on
// destruction — declare it AFTER the engine so it is destroyed first and the
// engine's join never waits on a held scan.
struct yield_gate {
    std::atomic<std::uint64_t> reached{0};
    std::atomic<std::uint64_t> allowed{0};

    std::function<void()> hook() {
        return [this] {
            const auto n = reached.fetch_add(1) + 1;
            while (n > allowed.load()) std::this_thread::yield();
        };
    }
    void allow(std::uint64_t n) { allowed.store(n); }
    void release_all() { allowed.store(UINT64_MAX); }
    // The scan thread is parked by this gate, so nothing can tick: the
    // no-heartbeat stall limit names a scan that never got here.
    [[nodiscard]] bool wait_reached(std::uint64_t n) const {
        return coast_test::await_condition([&] { return reached.load() >= n; });
    }
};
struct gate_release {
    yield_gate& g;
    ~gate_release() { g.release_all(); }
};

fs::path temp_path(const std::string& name)
{
    const fs::path  p = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove(p, ec);
    return p;
}

// Stamp the uniform /metadata@type marker open_and_map keys on. These
// fixtures follow the observe convention (value + value__step_index).
void stamp_observe(hid_t fid)
{
    const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
    REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
    H5Gclose(md);
}

// Observe-convention file: /act f32[n] with act[i] = i % 977 (chunk 1024)
// + step index. Optionally a second gzip'd dataset /packed for poisoning.
fs::path make_file(hdf5::executor& exec, const std::string& name,
                   std::uint64_t n, bool with_packed)
{
    const fs::path p = temp_path(name);
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        stamp_observe(fid);
        const hsize_t dims[1]  = {n};
        const hsize_t chunk[1] = {1024};
        const hid_t   sp   = H5Screate_simple(1, dims, nullptr);
        const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        std::vector<float> vals(n);
        for (std::uint64_t i = 0; i < n; ++i) {
            vals[i] = static_cast<float>(i % 977);
        }
        const hid_t d = H5Dcreate2(fid, "act", H5T_NATIVE_FLOAT, sp,
                                   H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 vals.data());
        H5Dclose(d);
        const hid_t d2 = H5Dcreate2(fid, "act__step_index", H5T_NATIVE_UINT64,
                                    sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dclose(d2);

        if (with_packed) {
            const hid_t zdcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(zdcpl, 1, chunk);
            H5Pset_shuffle(zdcpl);
            H5Pset_deflate(zdcpl, 4);
            const hid_t z = H5Dcreate2(fid, "packed", H5T_NATIVE_FLOAT, sp,
                                       H5P_DEFAULT, zdcpl, H5P_DEFAULT);
            H5Dwrite(z, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     vals.data());
            H5Dclose(z);
            const hid_t z2 = H5Dcreate2(fid, "packed__step_index",
                                        H5T_NATIVE_UINT64, sp, H5P_DEFAULT,
                                        dcpl, H5P_DEFAULT);
            H5Dclose(z2);
            H5Pclose(zdcpl);
        }
        H5Pclose(dcpl);
        H5Sclose(sp);
        H5Fclose(fid);
    });
    return p;
}

// Flips one byte inside the first stored (compressed) chunk of /packed —
// gzip'd chunks carry checksums via the filter pipeline, so reads fail
// per-dataset (the T3 "bit flip in compressed chunk data" class).
void poison_packed(hdf5::executor& exec, const fs::path& p)
{
    haddr_t  addr = HADDR_UNDEF;
    hsize_t  size = 0;
    exec.submit_sync([&] {
        const hid_t fid = H5Fopen(p.string().c_str(), H5F_ACC_RDONLY,
                                  H5P_DEFAULT);
        REQUIRE(fid >= 0);
        const hid_t d = H5Dopen2(fid, "packed", H5P_DEFAULT);
        REQUIRE(d >= 0);
        hsize_t offset[1] = {0};
        REQUIRE(H5Dget_chunk_info_by_coord(d, offset, nullptr, &addr, &size) >= 0);
        H5Dclose(d);
        H5Fclose(fid);
    });
    REQUIRE(addr != HADDR_UNDEF);
    REQUIRE(size > 16);

    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    f.seekg(static_cast<std::streamoff>(addr + size / 2));
    char b = 0;
    f.read(&b, 1);
    b = static_cast<char>(b ^ 0x5a);
    f.seekp(static_cast<std::streamoff>(addr + size / 2));
    f.write(&b, 1);
}

// Consume-poll helper: demand_published only re-produces after a consume,
// so the poll itself drives publication.
template <class Consume>
auto latest(Consume&& consume, decltype(consume())& last)
{
    if (auto s = consume()) last = std::move(s);
    return last;
}

} // namespace

TEST_CASE("scan_engine: open -> auto stats scan -> converged correct stats")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_basic.h5", 100'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::file_snapshot>  file;
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open && stats && stats->scan_complete;
    }));

    CHECK(file->model.family == h5b::file_family::observe);
    REQUIRE(stats->per_dataset.size() == 1);
    const auto& e = stats->per_dataset[0];
    CHECK(e.state == h5b::op_state::done);
    CHECK(e.stats.count == 100'000);
    CHECK(e.stats.min == 0.0);
    CHECK(e.stats.max == 976.0);
    CHECK(e.stats.nan_count == 0);
    // mean of i%977 over 100k values: 102 full cycles (sum 976*977/2 each)
    // + remainder 0..345 -> 48'690'837 / 100'000.
    CHECK(e.stats.mean == doctest::Approx(486.90837).epsilon(1e-9));

    // Log carries the completion line. It is written right after the
    // scan_complete snapshot is published, so it is awaited, not assumed to be
    // there already.
    bool saw_complete = false;
    CHECK(eventually(engine, [&] {
        for (const auto& l : engine.drain_log()) {
            if (l.message.find("stats scan complete") != std::string::npos) {
                saw_complete = true;
            }
        }
        return saw_complete;
    }));

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: poisoned dataset is contained, siblings scan to done")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_poison.h5", 100'000, true);
    poison_packed(exec, p);

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::file_snapshot>  file;
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open && stats && stats->scan_complete;
    }));

    REQUIRE(stats->per_dataset.size() == 2); // act + packed
    std::size_t done = 0, failed = 0;
    for (std::size_t i = 0; i < stats->per_dataset.size(); ++i) {
        const auto& e = stats->per_dataset[i];
        if (e.state == h5b::op_state::done) {
            ++done;
            CHECK(e.stats.count == 100'000); // the healthy sibling is complete
        }
        if (e.state == h5b::op_state::failed) {
            ++failed;
            CHECK_FALSE(e.error_message.empty());
        }
    }
    CHECK(done == 1);
    CHECK(failed == 1);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: fetch lane serves table blocks; misses are deduplicated")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_fetch.h5", 50'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && stats->scan_complete;
    }));

    // Miss -> request -> eventually served from the cache.
    CHECK(engine.try_get_block(0, 3) == nullptr);
    engine.request_block(0, 3);
    engine.request_block(0, 3); // dedupe: second request collapses
    REQUIRE(eventually(engine, [&] { return engine.try_get_block(0, 3) != nullptr; }));

    const auto blk = engine.try_get_block(0, 3);
    REQUIRE(blk != nullptr);
    CHECK(blk->first_row == 3 * 1024); // chunk-aligned block rows
    CHECK(blk->n_rows == 1024);
    const auto* f = reinterpret_cast<const float*>(blk->bytes.data());
    CHECK(f[0] == static_cast<float>((3 * 1024) % 977));

    // Out-of-range block: logged once, never crashes, stays absent.
    engine.request_block(0, 1'000'000);
    REQUIRE(eventually(engine, [&] {
        for (const auto& l : engine.drain_log()) {
            if (l.message.find("table fetch failed") != std::string::npos) {
                return true;
            }
        }
        return false;
    }));
    CHECK(engine.try_get_block(0, 1'000'000) == nullptr);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: open failure publishes error; engine stays usable")
{
    hdf5::executor exec;
    const fs::path bogus = temp_path("h5b_eng_bogus.h5");
    {
        std::ofstream out(bogus);
        out << "not hdf5";
    }

    h5b::scan_engine engine(exec);
    engine.open_file(bogus);

    std::shared_ptr<const h5b::file_snapshot> file;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        return file && !file->error_message.empty();
    }));
    CHECK_FALSE(file->open);

    // The same engine then opens a good file.
    const fs::path good = make_file(exec, "h5b_eng_good.h5", 10'000, false);
    engine.open_file(good);
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open && stats && stats->scan_complete;
    }));
    CHECK(stats->per_dataset[0].stats.count == 10'000);

    std::error_code ec;
    fs::remove(bogus, ec);
    fs::remove(good, ec);
}

TEST_CASE("scan_engine: per-column histograms — unfiltered, filtered, width cap")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_hist.h5", 100'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && stats->scan_complete;
    }));

    // Unfiltered: exact totals and bin contents for the i % 977 pattern.
    engine.compute_histograms(0, {});
    std::shared_ptr<const h5b::histogram_snapshot> hist;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_histograms(); }, hist);
        return hist && hist->state == h5b::op_state::done;
    }));
    CHECK(hist->dataset == 0);
    CHECK(hist->filter_desc == "unfiltered");
    CHECK(hist->rows_scanned == 100'000);
    CHECK(hist->rows_matched == 100'000);
    REQUIRE(hist->columns.size() == 1);
    const auto& h = hist->columns[0];
    CHECK(h.total() == 100'000);
    CHECK(h.lo() == 0.0);
    CHECK(h.hi() == 1024.0); // range doubled up from [0,1) to cover 976
    // bin b counts values {2b, 2b+1}; v <= 345 occurs 103 times, else 102.
    CHECK(h.bins()[0] == 206);   // v 0,1
    CHECK(h.bins()[200] == 204); // v 400,401
    CHECK(h.bins()[488] == 102); // v 976 only (977 never occurs)
    CHECK(h.bins()[489] == 0);

    // Filtered: col 0 > 500 -> 476 matches per full 977-cycle, none in the
    // remainder 0..345 -> 102 * 476 = 48'552.
    h5b::row_filter f;
    f.conditions = {{0, h5b::filter_op::gt, 500.0}};
    engine.compute_histograms(0, f);
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_histograms(); }, hist);
        return hist && hist->state == h5b::op_state::done &&
               hist->filter_desc != "unfiltered";
    }));
    CHECK(hist->rows_matched == 48'552);
    CHECK(hist->columns[0].total() == 48'552);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: row-index materialization for the filtered table view")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_ridx.h5", 100'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && stats->scan_complete;
    }));

    // v == 975 or 976 within rows [0, 2000): rows 975, 976, 1952, 1953.
    h5b::row_filter f;
    f.row_end    = 2000;
    f.conditions = {{0, h5b::filter_op::ge, 975.0}};
    engine.compute_row_index(0, f);

    std::shared_ptr<const h5b::row_index_snapshot> ridx;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_row_index(); }, ridx);
        return ridx && ridx->state == h5b::op_state::done;
    }));
    CHECK(ridx->rows_scanned == 100'000);
    CHECK(ridx->rows_matched == 4);
    CHECK(ridx->rows == std::vector<std::uint64_t>{975, 976, 1952, 1953});
    CHECK(ridx->filter_desc.find("col 0 >= 975") != std::string::npos);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: series decimation — exact buckets, range + filter")
{
    hdf5::executor exec;
    // 4096 rows with v = i % 977: series over 2048 buckets -> each bucket
    // covers exactly 2 rows (2b, 2b+1) with closed-form min/max/sum.
    const fs::path p = make_file(exec, "h5b_eng_series.h5", 4096, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && stats->scan_complete;
    }));

    engine.compute_series(0, 0, {});
    std::shared_ptr<const h5b::series_snapshot> ser;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_series(); }, ser);
        return ser && ser->state == h5b::op_state::done;
    }));
    CHECK(ser->rows_matched == 4096);
    REQUIRE(ser->buckets.size() == 2048);
    for (const std::size_t b : {std::size_t{0}, std::size_t{487},
                                std::size_t{488}, std::size_t{2047}}) {
        const double v0 = static_cast<double>((2 * b) % 977);
        const double v1 = static_cast<double>((2 * b + 1) % 977);
        CHECK(ser->buckets[b].count == 2);
        CHECK(ser->buckets[b].min == std::min(v0, v1));
        CHECK(ser->buckets[b].max == std::max(v0, v1));
        CHECK(ser->buckets[b].sum == v0 + v1);
    }

    // Row range bounds the scan itself; a condition empties some buckets.
    h5b::row_filter f;
    f.row_begin  = 100;
    f.row_end    = 200;
    f.conditions = {{0, h5b::filter_op::lt, 150.0}}; // rows 100..149 match
    engine.compute_series(0, 0, f);
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_series(); }, ser);
        return ser && ser->state == h5b::op_state::done &&
               ser->row_begin == 100;
    }));
    CHECK(ser->row_end == 200);
    CHECK(ser->rows_scanned == 100);
    CHECK(ser->rows_matched == 50);
    REQUIRE(ser->buckets.size() == 100); // span 100 < 2048 -> 1 row/bucket
    CHECK(ser->buckets[0].count == 1);   // row 100, v=100
    CHECK(ser->buckets[0].min == 100.0);
    CHECK(ser->buckets[49].count == 1);  // row 149, v=149
    CHECK(ser->buckets[50].count == 0);  // row 150 filtered out
    CHECK(ser->buckets[99].count == 0);

    // Out-of-range column fails cleanly.
    engine.compute_series(0, 5, {});
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_series(); }, ser);
        return ser && ser->state == h5b::op_state::failed;
    }));
    CHECK(ser->error_message.find("out of range") != std::string::npos);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: histogram width cap refuses with a logged failure")
{
    hdf5::executor exec;
    const fs::path p = temp_path("h5b_eng_wide.h5");
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        stamp_observe(fid);
        const hsize_t dims[2]  = {4, 16'385}; // one over the cap
        const hsize_t sdims[1] = {4};
        const hid_t   sp  = H5Screate_simple(2, dims, nullptr);
        const hid_t   ssp = H5Screate_simple(1, sdims, nullptr);
        std::vector<float> vals(4 * 16'385, 1.0f);
        const hid_t d = H5Dcreate2(fid, "wide", H5T_NATIVE_FLOAT, sp,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 vals.data());
        H5Dclose(d);
        const hid_t d2 = H5Dcreate2(fid, "wide__step_index",
                                    H5T_NATIVE_UINT64, ssp, H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(d2);
        H5Sclose(ssp);
        H5Sclose(sp);
        H5Fclose(fid);
    });

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && stats->scan_complete;
    }));

    engine.compute_histograms(0, {});
    std::shared_ptr<const h5b::histogram_snapshot> hist;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_histograms(); }, hist);
        return hist && hist->state == h5b::op_state::failed;
    }));
    CHECK(hist->error_message.find("capped") != std::string::npos);
    CHECK(hist->columns.empty());

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: cancel and session supersede leave a sane, responsive engine")
{
    hdf5::executor exec;
    const fs::path big   = make_file(exec, "h5b_eng_big.h5", 3'000'000, true);
    const fs::path small = make_file(exec, "h5b_eng_small.h5", 5'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(big);
    engine.cancel_analysis(); // may land mid-scan or after completion

    // Supersede: a second open aborts/queues past whatever the first is doing.
    engine.open_file(small);

    std::shared_ptr<const h5b::file_snapshot>  file;
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open &&
               file->model.path.filename() == "h5b_eng_small.h5" && stats &&
               stats->scan_complete;
    }));
    REQUIRE(stats->per_dataset.size() == 1);
    CHECK(stats->per_dataset[0].state == h5b::op_state::done);
    CHECK(stats->per_dataset[0].stats.count == 5'000);

    std::error_code ec;
    fs::remove(big, ec);
    fs::remove(small, ec);
}

// Deterministic cancellation: the scan thread is held at the first yield
// point of the auto stats scan (the open command is dequeued, so the cancel
// flag is no longer reset), the cancel is issued there, and the scan is
// released. That yield point must observe it: dataset 0 is cancelled before
// its first batch, and the whole transparent sequence aborts, leaving
// dataset 1 pending.
TEST_CASE("scan_engine: cancel interrupts the auto stats scan (whole sequence)")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_cancel.h5", 1'000'000, true);

    yield_gate       gate;
    h5b::scan_engine engine(exec, gate.hook());
    gate_release     release{gate};
    engine.open_file(p);

    REQUIRE(gate.wait_reached(1));   // the stats scan is running, held at yield 1
    engine.cancel_analysis();
    gate.release_all();

    // The wait ends on either outcome: dataset 0 cancelled, or (a cancel that
    // did not take) the scan run to completion. The checks below then say
    // which, at once, instead of a wait that only the stall limit ends.
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_stats(); }, stats);
        return stats && ((!stats->per_dataset.empty() &&
                          stats->per_dataset[0].state == h5b::op_state::cancelled) ||
                         stats->scan_complete);
    }));
    REQUIRE(stats->per_dataset.size() == 2);
    CHECK(stats->per_dataset[0].state == h5b::op_state::cancelled);
    CHECK(stats->per_dataset[0].rows_scanned == 0);            // cancelled at yield 1
    CHECK(stats->per_dataset[1].state == h5b::op_state::pending); // sequence aborted
    CHECK_FALSE(stats->scan_complete);

    // Responsiveness: a fresh open resets the cancel flag at dequeue and
    // scans to completion, uncancelled.
    const fs::path small = make_file(exec, "h5b_eng_cancel2.h5", 5'000, false);
    engine.open_file(small);
    std::shared_ptr<const h5b::file_snapshot> file;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open &&
               file->model.path.filename() == "h5b_eng_cancel2.h5" && stats &&
               stats->scan_complete;
    }));
    REQUIRE(stats->per_dataset.size() == 1);
    CHECK(stats->per_dataset[0].state == h5b::op_state::done);
    CHECK(stats->per_dataset[0].stats.count == 5'000);

    std::error_code ec;
    fs::remove(p, ec);
    fs::remove(small, ec);
}

// Two-lane servicing (D9a): a table block requested while the auto stats
// scan is running is served from the fetch lane before the scan finishes.
// The scan is held at yield point 1 while the block is requested, allowed
// exactly one batch (after which the scan loop services the fetch queue), and
// held again at yield point 2 — so the scan is provably still in progress
// when the block is checked, independent of how fast scan and fetch are.
TEST_CASE("scan_engine: fetch lane serves a block during an active scan")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_twolane.h5", 20'000'000, false);

    yield_gate       gate;
    h5b::scan_engine engine(exec, gate.hook());
    gate_release     release{gate};
    engine.open_file(p);

    REQUIRE(gate.wait_reached(1));   // scan running (model mapped), held at yield 1

    constexpr std::uint64_t target_block = 15'000; // mid-dataset, uncached
    engine.request_block(0, target_block);
    gate.allow(1);                   // one batch, then the loop services fetches
    REQUIRE(gate.wait_reached(2));   // held again: the scan is not finished

    std::shared_ptr<const h5b::stats_snapshot> stats;
    latest([&] { return engine.consume_stats(); }, stats);
    const bool served_during_scan =
        engine.try_get_block(0, target_block) != nullptr && !(stats && stats->scan_complete);
    gate.release_all();
    CHECK(served_during_scan); // the fetch did not wait for scan completion

    const auto blk = engine.try_get_block(0, target_block);
    REQUIRE(blk != nullptr);
    CHECK(blk->first_row == target_block * 1024);
    const auto* f = reinterpret_cast<const float*>(blk->bytes.data());
    CHECK(f[0] == static_cast<float>((target_block * 1024) % 977));

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("scan_engine: close_file tears down state and the engine reopens")
{
    hdf5::executor exec;
    const fs::path p = make_file(exec, "h5b_eng_close.h5", 5'000, false);

    h5b::scan_engine engine(exec);
    engine.open_file(p);

    std::shared_ptr<const h5b::file_snapshot>  file;
    std::shared_ptr<const h5b::stats_snapshot> stats;
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open && stats && stats->scan_complete;
    }));

    // Close: the file snapshot goes closed and the stats table is cleared.
    engine.close_file();
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && !file->open && stats && stats->per_dataset.empty();
    }));
    CHECK_FALSE(stats->scan_complete);

    // Reopen the same path: a fresh session scans to completion.
    engine.open_file(p);
    REQUIRE(eventually(engine, [&] {
        latest([&] { return engine.consume_file(); }, file);
        latest([&] { return engine.consume_stats(); }, stats);
        return file && file->open && stats && stats->scan_complete;
    }));
    REQUIRE(stats->per_dataset.size() == 1);
    CHECK(stats->per_dataset[0].state == h5b::op_state::done);
    CHECK(stats->per_dataset[0].stats.count == 5'000);

    std::error_code ec;
    fs::remove(p, ec);
}
