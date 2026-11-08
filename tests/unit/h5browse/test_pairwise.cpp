// h5browse pairwise — sampled + tiled full run through the real engine.
//
// Ground truths are closed-form: identical rows give cosine == 1 for every
// pair; rows (r, r) give Euclidean distance sqrt(2)|i-j|; a full run over
// N rows must produce a bin-identical histogram to a "sampled" run with
// k >= N (same selection, different code path), including across tile
// boundaries (N > 4096 forces multi-tile).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/scan_engine.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include "await_progress.h"

#include <hdf5.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Wait on the engine's own progress, not on a deadline (CLAUDE.md "Testing").
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

// Observe-convention file with a 2-wide f32 dataset "pts": row r = (v, v)
// where v = r (identical == false) or v = 1 (identical == true).
fs::path make_pts_file(hdf5::executor& exec, const std::string& name,
                       std::uint64_t n, bool identical)
{
    const fs::path  p = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove(p, ec);
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        // Uniform /metadata@type stamp open_and_map keys on (observe convention).
        {
            const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
            REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
            REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
            H5Gclose(md);
        }
        // Chunk must not exceed a fixed dimension: H5Dcreate2 refuses it.
        const hsize_t dims[2]  = {n, 2};
        const hsize_t chunk[2] = {std::min<hsize_t>(n, 256), 2};
        const hsize_t sdims[1] = {n};
        const hid_t   sp   = H5Screate_simple(2, dims, nullptr);
        const hid_t   ssp  = H5Screate_simple(1, sdims, nullptr);
        const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, chunk);
        std::vector<float> vals(n * 2);
        for (std::uint64_t r = 0; r < n; ++r) {
            const float v   = identical ? 1.0f : static_cast<float>(r);
            vals[r * 2]     = v;
            vals[r * 2 + 1] = v;
        }
        const hid_t d = H5Dcreate2(fid, "pts", H5T_NATIVE_FLOAT, sp,
                                   H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 vals.data());
        H5Dclose(d);
        const hid_t d2 = H5Dcreate2(fid, "pts__step_index", H5T_NATIVE_UINT64,
                                    ssp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(d2);
        H5Pclose(dcpl);
        H5Sclose(ssp);
        H5Sclose(sp);
        H5Fclose(fid);
    });
    return p;
}

struct engine_session {
    h5b::scan_engine                            engine;
    std::shared_ptr<const h5b::stats_snapshot>  stats;
    std::shared_ptr<const h5b::pairwise_snapshot> pair;

    explicit engine_session(hdf5::executor& exec) : engine(exec) {}

    void open_and_wait(const fs::path& p)
    {
        engine.open_file(p);
        REQUIRE(eventually(engine, [&] {
            if (auto s = engine.consume_stats()) stats = std::move(s);
            return stats && stats->scan_complete;
        }));
    }

    const h5b::pairwise_snapshot& pairwise_and_wait(std::uint32_t dataset,
                                                    h5b::pairwise_request req)
    {
        pair.reset();
        engine.compute_pairwise(dataset, req);
        REQUIRE(eventually(engine, [&] {
            if (auto s = engine.consume_pairwise()) pair = std::move(s);
            return pair && (pair->state == h5b::op_state::done ||
                            pair->state == h5b::op_state::failed);
        }));
        return *pair;
    }
};

} // namespace

TEST_CASE("pairwise: identical rows -> every cosine similarity is exactly 1")
{
    hdf5::executor exec;
    const fs::path p = make_pts_file(exec, "h5b_pw_ident.h5", 100, true);

    engine_session s(exec);
    s.open_and_wait(p);

    h5b::pairwise_request req;
    req.kind = coast::metric_kind::cosine_similarity;
    req.k    = 2000; // > 100 rows -> all rows selected
    const auto& r = s.pairwise_and_wait(0, req);

    CHECK(r.state == h5b::op_state::done);
    CHECK(r.rows_in_filter == 100);
    CHECK(r.pairs_total == 100 * 99 / 2);
    CHECK(r.pairs_done == r.pairs_total);
    CHECK(r.histogram.total() == r.pairs_total);
    // Fixed cosine domain [-1,1]; value 1.0 clamps into the top bin.
    CHECK(r.histogram.lo() == -1.0);
    CHECK(r.histogram.hi() == 1.0);
    CHECK(r.histogram.bins().back() == r.pairs_total);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("pairwise: Euclidean distances sqrt(2)|i-j| land in exact adaptive bins")
{
    hdf5::executor exec;
    const fs::path p = make_pts_file(exec, "h5b_pw_dist.h5", 10, false);

    engine_session s(exec);
    s.open_and_wait(p);

    h5b::pairwise_request req;
    req.kind     = coast::metric_kind::minkowski_distance;
    req.params.p = 2.0f;
    req.k        = 10;
    const auto& r = s.pairwise_and_wait(0, req);

    CHECK(r.state == h5b::op_state::done);
    CHECK(r.pairs_total == 45);
    CHECK(r.histogram.total() == 45);
    CHECK(r.histogram.lo() == 0.0); // distances anchor at 0 (nonneg form)
    // Max distance sqrt(2)*9 = 12.7 -> range [0, 16).
    CHECK(r.histogram.hi() == 16.0);
    // |i-j| = d occurs (10-d) times, at distance sqrt(2)d; bin = v/(16/4096).
    for (int dd = 1; dd <= 9; ++dd) {
        const double v = std::sqrt(2.0) * dd;
        const auto   bin =
            static_cast<std::size_t>(v / (16.0 / h5b::pairwise_base_bins));
        CHECK(r.histogram.bins()[bin] == static_cast<std::uint64_t>(10 - dd));
    }

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("pairwise: filter restricts the population; k caps the sample")
{
    hdf5::executor exec;
    const fs::path p = make_pts_file(exec, "h5b_pw_filt.h5", 1000, false);

    engine_session s(exec);
    s.open_and_wait(p);

    h5b::pairwise_request req;
    req.kind     = coast::metric_kind::minkowski_distance;
    req.params.p = 2.0f;
    req.k        = 50;
    req.filter.conditions = {{0, h5b::filter_op::ge, 900.0}}; // rows 900..999
    const auto& r = s.pairwise_and_wait(0, req);

    CHECK(r.state == h5b::op_state::done);
    CHECK(r.rows_in_filter == 100);
    CHECK(r.pairs_total == 50 * 49 / 2); // k=50 sampled from 100
    CHECK(r.histogram.total() == r.pairs_total);
    // All sampled rows are (v,v) with v >= 900: max distance sqrt(2)*99
    // -> adaptive range stays within [0, 256).
    CHECK(r.histogram.hi() <= 256.0);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("pairwise: multi-tile full run is bin-identical to single-selection run")
{
    hdf5::executor exec;
    // 5000 rows > pairwise_tile_rows -> 2 tiles; ~12.5M pairs of width-2
    // rows stay CPU-cheap.
    const fs::path p = make_pts_file(exec, "h5b_pw_full.h5", 5000, false);

    engine_session s(exec);
    s.open_and_wait(p);

    h5b::pairwise_request full;
    full.kind     = coast::metric_kind::minkowski_distance;
    full.params.p = 1.0f;
    full.full_run = true;
    const auto rfull = s.pairwise_and_wait(0, full); // copy snapshot

    CHECK(rfull.state == h5b::op_state::done);
    const std::uint64_t expect_pairs = 5000ull * 4999 / 2;
    CHECK(rfull.pairs_total == expect_pairs);
    CHECK(rfull.pairs_done == expect_pairs);
    CHECK(rfull.histogram.total() == expect_pairs);

    // Same pairs via the sampled path (k >= N selects everything, one
    // selection vector instead of implicit iota tiles).
    h5b::pairwise_request all;
    all.kind     = coast::metric_kind::minkowski_distance;
    all.params.p = 1.0f;
    all.k        = 5000;
    const auto& rall = s.pairwise_and_wait(0, all);

    CHECK(rall.state == h5b::op_state::done);
    REQUIRE(rall.histogram.n_bins() == rfull.histogram.n_bins());
    CHECK(rall.histogram.lo() == rfull.histogram.lo());
    CHECK(rall.histogram.hi() == rfull.histogram.hi());
    const auto ba = rall.histogram.bins(), bf = rfull.histogram.bins();
    for (std::size_t i = 0; i < ba.size(); ++i) CHECK(ba[i] == bf[i]);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("pairwise: fewer than two selected rows completes empty, no crash")
{
    hdf5::executor exec;
    const fs::path p = make_pts_file(exec, "h5b_pw_tiny.h5", 5, false);

    engine_session s(exec);
    s.open_and_wait(p);

    h5b::pairwise_request req;
    req.filter.conditions = {{0, h5b::filter_op::gt, 100.0}}; // matches nothing
    const auto& r = s.pairwise_and_wait(0, req);

    CHECK(r.state == h5b::op_state::done);
    CHECK(r.rows_in_filter == 0);
    CHECK(r.pairs_total == 0);
    CHECK(r.histogram.total() == 0);

    std::error_code ec;
    fs::remove(p, ec);
}
