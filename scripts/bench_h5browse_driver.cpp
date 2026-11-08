// h5-browser G-suite benchmark driver (G1-G7 from .dew/docs/01-discover.md).
// Built and run by scripts/bench_h5browse.sh against libcoast_lib.a.
//
// Modes (one measurement per G-target, no voyages):
//   open FILE            G1: open_and_map wall time (cold cache)
//   scan FILE            G2: time-to-first-stats + sustained logical GB/s
//   rawread FILE         G2 denominator: plain pread sequential read GB/s
//   hist FILE DSIDX      G3: histogram pass + rebin timings
//   pairwise FILE DSIDX  G4: sampled k=2000 wall time (+ cosine, default p)
//   search FILE          G5: per-keystroke tree filter cost
//   corpus DIR           G6: open (+scan on success) every file; process
//                        survival = pass; prints per-file outcome
//   stall FILE           G7 proxy: max GUI-side consume/try_get latency
//                        while a full scan of FILE runs
//
// Cold cache: posix_fadvise(DONTNEED) over the whole file before timed
// opens (same technique as the T1 experiments).

#include "h5browse/file_model.h"
#include "h5browse/scan_engine.h"
#include "hdf5/executor.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using clk    = std::chrono::steady_clock;

namespace {

double secs_since(clk::time_point t0)
{
    return std::chrono::duration<double>(clk::now() - t0).count();
}

void drop_cache(const fs::path& p)
{
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return;
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

std::uint64_t logical_bytes(const h5b::file_model& m)
{
    std::uint64_t total = 0;
    for (const auto& d : m.datasets) {
        total += d.rows * d.row_width * h5b::scalar_size(d.dtype);
    }
    return total;
}

// Poll helper: consume until pred, with deadline; returns wall seconds.
template <class Pred>
double wait_for(h5b::scan_engine& eng,
                std::shared_ptr<const h5b::stats_snapshot>& stats, Pred pred,
                double deadline_s = 600.0)
{
    const auto t0 = clk::now();
    while (secs_since(t0) < deadline_s) {
        if (auto s = eng.consume_stats()) stats = std::move(s);
        if (stats && pred(*stats)) return secs_since(t0);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return -1.0;
}

int mode_open(const fs::path& file)
{
    drop_cache(file);
    hdf5::executor exec;
    const auto     t0 = clk::now();
    const auto     m  = h5b::open_and_map(exec, file);
    const double   dt = secs_since(t0);
    if (!m) {
        std::printf("open FAILED: %s\n", m.error().message.c_str());
        return 1;
    }
    std::printf("G1 open_and_map: %.3f s  (%zu datasets, family %s)\n", dt,
                m->datasets.size(), h5b::file_family_name(m->family));
    return 0;
}

int mode_scan(const fs::path& file)
{
    drop_cache(file);
    hdf5::executor   exec;
    h5b::scan_engine eng(exec);

    const auto t0 = clk::now();
    eng.open_file(file);

    std::shared_ptr<const h5b::stats_snapshot> stats;
    const double first = wait_for(eng, stats, [](const auto& s) {
        for (const auto& e : s.per_dataset) {
            if (e.stats.count > 0) return true;
        }
        return false;
    });
    const double total = wait_for(
        eng, stats, [](const auto& s) { return s.scan_complete; });
    const double total_wall = secs_since(t0);

    // Logical bytes actually scanned (skip failed datasets).
    auto m = eng.consume_file();
    std::uint64_t bytes = m && m->open ? logical_bytes(m->model) : 0;
    std::printf("G2 first stats: %.3f s\n", first);
    std::printf("G2 full scan:   %.3f s  (%.2f GB logical, %.2f GB/s)\n",
                total_wall, static_cast<double>(bytes) / 1e9,
                static_cast<double>(bytes) / 1e9 / total_wall);
    return total < 0 ? 1 : 0;
}

int mode_rawread(const fs::path& file)
{
    drop_cache(file);
    const int fd = ::open(file.c_str(), O_RDONLY);
    if (fd < 0) return 1;
    std::vector<char> buf(8u << 20);
    const auto        t0    = clk::now();
    std::uint64_t     total = 0;
    while (true) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n <= 0) break;
        total += static_cast<std::uint64_t>(n);
    }
    ::close(fd);
    const double dt = secs_since(t0);
    std::printf("raw sequential read: %.3f s  (%.2f GB, %.2f GB/s)\n", dt,
                static_cast<double>(total) / 1e9,
                static_cast<double>(total) / 1e9 / dt);
    return 0;
}

int mode_hist(const fs::path& file, std::uint32_t ds)
{
    hdf5::executor   exec;
    h5b::scan_engine eng(exec);
    eng.open_file(file);
    std::shared_ptr<const h5b::stats_snapshot> stats;
    if (wait_for(eng, stats, [](const auto& s) { return s.scan_complete; }) < 0) {
        return 1;
    }

    const auto t0 = clk::now();
    eng.compute_histograms(ds, {});
    std::shared_ptr<const h5b::histogram_snapshot> hist;
    while (true) {
        if (auto h = eng.consume_histograms()) hist = std::move(h);
        if (hist && (hist->state == h5b::op_state::done ||
                     hist->state == h5b::op_state::failed)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    std::printf("G3 histogram pass: %.3f s (%zu columns, %" PRIu64 " rows)\n",
                secs_since(t0), hist->columns.size(), hist->rows_scanned);
    if (hist->state != h5b::op_state::done) {
        std::printf("histogram FAILED: %s\n", hist->error_message.c_str());
        return 1;
    }

    // G3 core claim: rebin is re-aggregation, no rescan.
    const auto&  h  = hist->columns[0];
    const auto   t1 = clk::now();
    std::uint64_t sink = 0;
    for (int rep = 0; rep < 1000; ++rep) {
        for (std::uint32_t m : {8u, 16u, 32u, 64u, 128u, 256u}) {
            sink += h.rebin(m)[0];
        }
    }
    std::printf("G3 rebin (6 widths x1000): %.3f ms total -> %.4f ms per "
                "rebin [sink %" PRIu64 "]\n",
                secs_since(t1) * 1e3, secs_since(t1) * 1e3 / 6000.0, sink);
    return 0;
}

int mode_pairwise(const fs::path& file, std::uint32_t ds)
{
    hdf5::executor   exec;
    h5b::scan_engine eng(exec);
    eng.open_file(file);
    std::shared_ptr<const h5b::stats_snapshot> stats;
    if (wait_for(eng, stats, [](const auto& s) { return s.scan_complete; }) < 0) {
        return 1;
    }
    auto m = eng.consume_file();
    std::printf("dataset %u: %" PRIu64 " rows x %" PRIu64 " cols\n", ds,
                m->model.datasets[ds].rows, m->model.datasets[ds].row_width);

    h5b::pairwise_request req; // cosine, k=2000, seed 42
    const auto            t0 = clk::now();
    eng.compute_pairwise(ds, req);
    std::shared_ptr<const h5b::pairwise_snapshot> pair;
    while (true) {
        if (auto p = eng.consume_pairwise()) pair = std::move(p);
        if (pair && (pair->state == h5b::op_state::done ||
                     pair->state == h5b::op_state::failed)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    std::printf("G4 pairwise sampled k=2000 cosine: %.3f s (%" PRIu64
                " pairs, %" PRIu64 " in histogram)\n",
                secs_since(t0), pair->pairs_done, pair->histogram.total());
    return pair->state == h5b::op_state::done ? 0 : 1;
}

// Same matching logic as the GUI tree filter.
bool tree_matches(const h5b::tree_node& n, const std::string& needle)
{
    std::string lower(n.name);
    std::ranges::transform(lower, lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower.find(needle) != std::string::npos) return true;
    for (const auto& c : n.children) {
        if (tree_matches(c, needle)) return true;
    }
    return false;
}

int mode_search(const fs::path& file)
{
    hdf5::executor exec;
    const auto     m = h5b::open_and_map(exec, file);
    if (!m) return 1;
    const char* needles[] = {"n", "ne", "neu", "neur", "act", "activity",
                             "zzz", "[12]"};
    const auto  t0        = clk::now();
    int         visible   = 0;
    for (int rep = 0; rep < 100; ++rep) {
        for (const char* nd : needles) {
            for (const auto& c : m->root.children) {
                visible += tree_matches(c, nd) ? 1 : 0;
            }
        }
    }
    const double per_keystroke = secs_since(t0) / (100.0 * 8.0);
    std::printf("G5 tree filter: %.3f ms per keystroke (%zu datasets) "
                "[visible %d]\n",
                per_keystroke * 1e3, m->datasets.size(), visible);
    return 0;
}

int mode_corpus(const fs::path& dir)
{
    hdf5::executor exec;
    int            files = 0, opened = 0, rejected = 0;
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file()) paths.push_back(e.path());
    }
    std::ranges::sort(paths);
    for (const auto& p : paths) {
        ++files;
        const auto m = h5b::open_and_map(exec, p);
        if (m) {
            // Opened: scan it too — G6 covers analysis, not just open.
            h5b::scan_engine eng(exec);
            eng.open_file(p);
            std::shared_ptr<const h5b::stats_snapshot> stats;
            const double t = wait_for(
                eng, stats, [](const auto& s) { return s.scan_complete; },
                120.0);
            std::uint64_t failed_ds = 0;
            if (stats) {
                for (const auto& e2 : stats->per_dataset) {
                    if (e2.state == h5b::op_state::failed) ++failed_ds;
                }
            }
            std::printf("  OPEN+SCAN %-40s (%.1f s, %" PRIu64
                        " damaged datasets contained)\n",
                        p.filename().string().c_str(), t, failed_ds);
            ++opened;
        } else {
            std::printf("  REJECT    %-40s [%s] %s\n",
                        p.filename().string().c_str(),
                        h5b::error_category_name(m.error().category),
                        m.error().message.c_str());
            if (m.error().message.empty()) return 2; // non-actionable
            ++rejected;
        }
    }
    std::printf("G6 corpus: %d files, %d opened, %d cleanly rejected, 0 "
                "crashes (process survived)\n",
                files, opened, rejected);
    return 0;
}

int mode_stall(const fs::path& file)
{
    drop_cache(file);
    hdf5::executor   exec;
    h5b::scan_engine eng(exec);
    eng.open_file(file);

    // Hammer the full GUI-thread per-frame surface while the scan runs;
    // record the worst single-call latency.
    double                                     worst_ms = 0.0;
    std::shared_ptr<const h5b::stats_snapshot> stats;
    const auto                                 t0 = clk::now();
    std::uint64_t calls = 0;
    while (secs_since(t0) < 600.0) {
        const auto c0 = clk::now();
        auto       f  = eng.consume_file();
        if (auto s = eng.consume_stats()) stats = std::move(s);
        auto h = eng.consume_histograms();
        auto p = eng.consume_pairwise();
        auto l = eng.drain_log();
        auto b = eng.try_get_block(0, calls % 64);
        eng.request_block(0, calls % 64);
        worst_ms = std::max(worst_ms, secs_since(c0) * 1e3);
        ++calls;
        if (stats && stats->scan_complete) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::printf("G7 GUI-side worst call latency during full scan: %.2f ms "
                "(%" PRIu64 " frames)\n",
                worst_ms, calls);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s open|scan|rawread|hist|pairwise|search|corpus|"
                     "stall PATH [DSIDX]\n",
                     argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const fs::path    path = argv[2];
    const auto        ds   = static_cast<std::uint32_t>(
        argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 0);

    if (mode == "open") return mode_open(path);
    if (mode == "scan") return mode_scan(path);
    if (mode == "rawread") return mode_rawread(path);
    if (mode == "hist") return mode_hist(path, ds);
    if (mode == "pairwise") return mode_pairwise(path, ds);
    if (mode == "search") return mode_search(path);
    if (mode == "corpus") return mode_corpus(path);
    if (mode == "stall") return mode_stall(path);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
