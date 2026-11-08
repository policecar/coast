// tests/unit/io/test_sample_sequencer.cpp
//
// Unit tests for dataset::sample_sequencer (include/io/sample_sequencer.h):
//
//   - SEQUENTIAL: identity order, correct change_interval hold, no RNG draws
//     (golden-safety: shuffle-off must reproduce the legacy advancement exactly).
//   - SHUFFLE: every index appears exactly once per epoch; consecutive epochs
//     differ; same seed -> same sequence, different seed -> different; sample
//     held for change_interval steps.
//   - save_state -> load_state mid-epoch reproduces the exact remaining sequence
//     (resume parity), in both modes, via the in-memory state tree.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "io/sample_sequencer.h"
#include "io/sequence_params.h"
#include "diag/diag.h"
#include "logging/mem_state_io.h"
#include "logging/rng_state_io.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <system_error>
#include <vector>

using dataset::sample_sequencer;

namespace fs = std::filesystem;

namespace {

// Drive a sequencer for `steps` process()-equivalent steps, recording the
// presented index at each step (current_index() then advance()).
std::vector<std::size_t> run(sample_sequencer& seq, std::size_t steps)
{
    std::vector<std::size_t> out;
    out.reserve(steps);
    for (std::size_t s = 0; s < steps; ++s) {
        out.push_back(seq.current_index());
        seq.advance();
    }
    return out;
}

fs::path write_manifest(std::string_view name, std::string_view text)
{
    const fs::path dir = fs::temp_directory_path() / "test_sample_sequencer";
    std::error_code ec;
    fs::create_directories(dir, ec);
    REQUIRE_FALSE(ec);

    const fs::path path = dir / name;
    std::ofstream out(path);
    REQUIRE(out.good());
    out << text;
    return path;
}

} // namespace

TEST_CASE("sequential mode is identity order with change_interval hold")
{
    constexpr std::size_t N  = 6;
    constexpr int         CI = 3;
    sample_sequencer seq(N, CI, /*shuffle=*/false, /*seed=*/42);

    // First sample held for CI steps, then the next, wrapping after N samples.
    const auto seen = run(seq, N * CI);

    for (std::size_t sample = 0; sample < N; ++sample) {
        for (int k = 0; k < CI; ++k) {
            CHECK(seen[sample * CI + k] == sample);
        }
    }
    // After exactly one epoch's worth of steps the cursor is back at 0, epoch 1.
    CHECK(seq.current_index() == 0);
    CHECK(seq.current_epoch() == 1);
}

TEST_CASE("sequential mode draws zero RNG values (golden-safety invariant)")
{
    // SEQUENTIAL must never advance the RNG, so a stepped sequencer's persisted
    // seq_rng stays byte-identical to a freshly constructed one's; any spurious
    // draw would desynchronise the two streams. Resume-parity tests can't catch
    // a stray draw (load_state restores the RNG verbatim), so compare the saved
    // RNG directly: equal state -> identical next draws.
    constexpr std::size_t N  = 12;
    constexpr int         CI = 2;

    sample_sequencer fresh(N, CI, /*shuffle=*/false, /*seed=*/321);
    sample_sequencer stepped(N, CI, /*shuffle=*/false, /*seed=*/321);
    run(stepped, N * CI * 3 + 7);   // several epochs, stopping mid-sample

    auto saved_rng = [](const sample_sequencer& s) {
        logging::mem_state_writer w;
        s.save_state(w);
        const auto tree = w.take_root();
        logging::mem_state_reader r(tree, logging::k_format_version);
        fast_rng_t rng(0);
        logging::read_rng(r, "seq_rng", rng, "test");
        return rng;
    };

    fast_rng_t rf = saved_rng(fresh);
    fast_rng_t rs = saved_rng(stepped);
    for (int i = 0; i < 4; ++i) {
        CHECK(rf() == rs());
    }
}

TEST_CASE("change_interval <= 0 holds a static frame and never advances")
{
    sample_sequencer seq(10, /*change_interval=*/0, /*shuffle=*/false, /*seed=*/1);
    for (int s = 0; s < 50; ++s) {
        CHECK(seq.current_index() == 0);
        seq.advance();
    }
    CHECK(seq.current_epoch() == 0);
}

TEST_CASE("index manifest order maps logical positions to source indices")
{
    sample_sequencer seq(10, 1, /*shuffle=*/false, /*seed=*/1,
                         std::vector<std::size_t>{3, 1, 4});
    const auto seen = run(seq, 6);
    CHECK(seen == std::vector<std::size_t>{3, 1, 4, 3, 1, 4});
    CHECK(seq.current_epoch() == 2);
}

TEST_CASE("shuffle mode shuffles within the index manifest only")
{
    const std::vector<std::size_t> manifest{8, 2, 5, 4};
    sample_sequencer seq(10, 1, /*shuffle=*/true, /*seed=*/7, manifest);
    const auto order = run(seq, manifest.size());
    CHECK(std::set<std::size_t>(order.begin(), order.end())
          == std::set<std::size_t>(manifest.begin(), manifest.end()));
}

TEST_CASE("index manifest rejects source indices outside the source dataset")
{
    CHECK_THROWS_AS(sample_sequencer(3, 1, /*shuffle=*/false, /*seed=*/1,
                                     std::vector<std::size_t>{0, 3}),
                    std::runtime_error);
}

TEST_CASE("index manifest rejects duplicate source indices")
{
    CHECK_THROWS_AS(sample_sequencer(5, 1, /*shuffle=*/false, /*seed=*/1,
                                     std::vector<std::size_t>{0, 2, 2}),
                    std::runtime_error);
}

TEST_CASE("load_index_manifest parses comments, commas, and whitespace")
{
    const fs::path manifest = write_manifest("valid.idx",
                                             "# validation subset\n"
                                             "3, 1\n"
                                             "4 7 # inline comment\n");
    CHECK(dataset::load_index_manifest(manifest.string(), 10)
          == std::vector<std::size_t>{3, 1, 4, 7});
}

TEST_CASE("ensure_manifest_readable fails fast on a bad path, skips empty paths")
{
    dataset::ensure_manifest_readable("");   // no manifest configured: no-op

    const fs::path ok = write_manifest("readable.idx", "0 1 2\n");
    dataset::ensure_manifest_readable(ok.string());

    CHECK_THROWS_AS(dataset::ensure_manifest_readable(
                        (ok.parent_path() / "missing.idx").string()),
                    std::runtime_error);
}

TEST_CASE("load_index_manifest rejects duplicate or out-of-range indices")
{
    auto load = [](const fs::path& path) {
        (void)dataset::load_index_manifest(path.string(), 10);
    };
    CHECK_THROWS_AS(load(write_manifest("duplicate.idx", "0\n2\n2\n")),
                    std::runtime_error);
    CHECK_THROWS_AS(load(write_manifest("out_of_range.idx", "0 10\n")),
                    std::runtime_error);
}

TEST_CASE("shuffle mode: each index appears exactly once per epoch")
{
    constexpr std::size_t N  = 50;
    constexpr int         CI = 1;
    sample_sequencer seq(N, CI, /*shuffle=*/true, /*seed=*/7);

    const auto epoch0 = run(seq, N);   // CI==1 -> one index per step
    std::set<std::size_t> uniq(epoch0.begin(), epoch0.end());
    CHECK(uniq.size() == N);
    CHECK(*uniq.begin() == 0);
    CHECK(*uniq.rbegin() == N - 1);

    // Next epoch is also a full permutation, and differs from the first.
    const auto epoch1 = run(seq, N);
    std::set<std::size_t> uniq1(epoch1.begin(), epoch1.end());
    CHECK(uniq1.size() == N);
    CHECK(epoch0 != epoch1);
    CHECK(seq.current_epoch() == 2);
}

TEST_CASE("shuffle mode: not the identity permutation")
{
    constexpr std::size_t N = 100;
    sample_sequencer seq(N, 1, /*shuffle=*/true, /*seed=*/123);
    const auto order = run(seq, N);
    std::vector<std::size_t> identity(N);
    for (std::size_t i = 0; i < N; ++i) identity[i] = i;
    CHECK(order != identity);   // astronomically unlikely to be identity
}

TEST_CASE("shuffle is seed-deterministic")
{
    constexpr std::size_t N = 40;
    sample_sequencer a(N, 1, true, 99);
    sample_sequencer b(N, 1, true, 99);
    sample_sequencer c(N, 1, true, 100);

    const auto sa = run(a, N);
    const auto sb = run(b, N);
    const auto sc = run(c, N);

    CHECK(sa == sb);   // same seed -> same sequence
    CHECK(sa != sc);   // different seed -> different sequence
}

TEST_CASE("shuffle mode holds each sample for change_interval steps")
{
    constexpr std::size_t N  = 8;
    constexpr int         CI = 4;
    sample_sequencer seq(N, CI, true, 5);

    const auto seen = run(seq, N * CI);
    // Within each block of CI steps the presented index is constant.
    for (std::size_t block = 0; block < N; ++block) {
        const std::size_t v = seen[block * CI];
        for (int k = 1; k < CI; ++k) {
            CHECK(seen[block * CI + k] == v);
        }
    }
}

TEST_CASE("resume parity: save/load mid-epoch reproduces the remaining sequence (sequential)")
{
    constexpr std::size_t N  = 20;
    constexpr int         CI = 2;

    // Reference run: advance partway, then keep going to the horizon.
    sample_sequencer ref(N, CI, false, 314);
    run(ref, 17);                       // burn 17 steps (mid-epoch)
    const auto tail_ref = run(ref, 40); // record the next 40 indices

    // Resume run: same prefix, save, build a fresh sequencer, load, continue.
    sample_sequencer src(N, CI, false, 314);
    run(src, 17);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(N, CI, false, 314);
    logging::mem_state_reader r(tree, logging::k_format_version);
    dst.load_state(r);
    const auto tail_dst = run(dst, 40);

    CHECK(tail_ref == tail_dst);
}

TEST_CASE("resume parity: save/load mid-epoch reproduces the remaining sequence (shuffle)")
{
    constexpr std::size_t N  = 25;
    constexpr int         CI = 3;

    // Burn enough steps to cross at least one epoch boundary so the resumed run
    // must regenerate a matching next-epoch permutation from the restored RNG.
    constexpr std::size_t burn = N * CI + 11;

    sample_sequencer ref(N, CI, true, 2718);
    run(ref, burn);
    const auto tail_ref = run(ref, 3 * N * CI);

    sample_sequencer src(N, CI, true, 2718);
    run(src, burn);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(N, CI, true, 2718);
    logging::mem_state_reader r(tree, logging::k_format_version);
    dst.load_state(r);
    const auto tail_dst = run(dst, 3 * N * CI);

    CHECK(tail_ref == tail_dst);
}

TEST_CASE("resume parity: manifest-backed sequencer (sequential)")
{
    const std::vector<std::size_t> manifest{9, 4, 0, 7, 2, 11, 5};
    constexpr int CI = 2;

    sample_sequencer ref(20, CI, false, 271, manifest);
    run(ref, 9);                        // burn 9 steps (mid-epoch)
    const auto tail_ref = run(ref, 30);

    sample_sequencer src(20, CI, false, 271, manifest);
    run(src, 9);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(20, CI, false, 271, manifest);
    logging::mem_state_reader r(tree, logging::k_format_version);
    dst.load_state(r);
    const auto tail_dst = run(dst, 30);

    CHECK(tail_ref == tail_dst);
    // The restored cursor still maps through the manifest: only manifest source
    // indices are ever presented after the load.
    for (const std::size_t idx : tail_dst) {
        CHECK(std::ranges::find(manifest, idx) != manifest.end());
    }
}

TEST_CASE("resume parity: manifest-backed sequencer across epochs (shuffle)")
{
    const std::vector<std::size_t> manifest{8, 2, 5, 4, 13, 1, 10};
    constexpr int CI = 3;
    // Cross an epoch boundary so the resumed run must regenerate a matching
    // next-epoch permutation over manifest positions from the restored RNG.
    const std::size_t burn = manifest.size() * CI + 5;

    sample_sequencer ref(14, CI, true, 1618, manifest);
    run(ref, burn);
    const auto tail_ref = run(ref, 3 * manifest.size() * CI);

    sample_sequencer src(14, CI, true, 1618, manifest);
    run(src, burn);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(14, CI, true, 1618, manifest);
    logging::mem_state_reader r(tree, logging::k_format_version);
    dst.load_state(r);
    const auto tail_dst = run(dst, 3 * manifest.size() * CI);

    CHECK(tail_ref == tail_dst);
    for (const std::size_t idx : tail_dst) {
        CHECK(std::ranges::find(manifest, idx) != manifest.end());
    }
}

TEST_CASE("legacy checkpoint cursor maps into the manifest's logical position")
{
    // Pre-sequencer checkpoints saved only cur_epoch/cur_idx/ci_cnt (cur_idx is
    // a *source* index); build one by hand with no seq_pos / seq_rng fields.
    logging::mem_state_writer w;
    w.write("cur_epoch", std::uint64_t{2});
    w.write("cur_idx", std::uint64_t{4});
    w.write("ci_cnt", std::uint64_t{0});
    const auto tree = w.take_root();

    sample_sequencer seq(10, 1, /*shuffle=*/false, /*seed=*/1,
                         std::vector<std::size_t>{3, 1, 4});
    logging::mem_state_reader r(tree, logging::k_format_version);
    seq.load_state(r);

    // Source index 4 sits at logical position 2; the sequence continues from
    // there and wraps back to the manifest start.
    const auto seen = run(seq, 3);
    CHECK(seen == std::vector<std::size_t>{4, 3, 1});
}

TEST_CASE("legacy checkpoint cursor outside the manifest fails fast")
{
    logging::mem_state_writer w;
    w.write("cur_epoch", std::uint64_t{0});
    w.write("cur_idx", std::uint64_t{7});   // not in {3, 1, 4}
    w.write("ci_cnt", std::uint64_t{0});
    const auto tree = w.take_root();

    sample_sequencer seq(10, 1, /*shuffle=*/false, /*seed=*/1,
                         std::vector<std::size_t>{3, 1, 4});
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_THROWS_AS(seq.load_state(r), std::runtime_error);
}

TEST_CASE("modern checkpoint loaded under a different manifest fails fast")
{
    // A checkpoint carries seq_pos (a logical position), not the manifest it was
    // saved against. Restoring it under a *different* manifest must not silently
    // resume at an unrelated sample -- the saved source index no longer maps to
    // the restored position, so load_state throws.
    const std::vector<std::size_t> saved_manifest{9, 4, 0, 7, 2};
    const std::vector<std::size_t> other_manifest{3, 8, 1, 6, 5};   // disjoint
    constexpr int CI = 1;

    sample_sequencer src(20, CI, /*shuffle=*/false, /*seed=*/5, saved_manifest);
    run(src, 3);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(20, CI, /*shuffle=*/false, /*seed=*/5, other_manifest);
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_THROWS_AS(dst.load_state(r), std::runtime_error);
}

// --- #238: a malformed saved permutation is refused before it is used --------
//
// load_state used to check only seq_perm's length. An entry past the dataset
// then became the presented index (no manifest) or indexed the manifest
// unchecked (SIGSEGV); duplicates silently presented a sample twice in one
// epoch (S.5.7). Each case below builds a checkpoint from a real save_state
// and corrupts seq_perm in one way.

namespace {

// A shuffle checkpoint of `seq` with seq_perm replaced by `perm`.
logging::mem_group with_perm(const sample_sequencer& seq, std::vector<std::uint64_t> perm)
{
    logging::mem_state_writer w;
    seq.save_state(w);
    auto tree = w.take_root();
    tree.fields["seq_perm"] = std::move(perm);
    return tree;
}

// What a sequencer presents from now on, without disturbing it.
std::vector<std::size_t> peek(sample_sequencer seq, std::size_t steps)
{
    return run(seq, steps);
}

} // namespace

TEST_CASE("load_state refuses a seq_perm entry out of range (no manifest)")
{
    sample_sequencer seq(4, 1, /*shuffle=*/true, /*seed=*/0);
    const auto tree = with_perm(seq, {9999999, 1, 2, 3});
    const auto before = peek(seq, 12);
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_THROWS_WITH_AS(seq.load_state(r),
                         doctest::Contains("the checkpoint is corrupt"),
                         std::runtime_error);
    // Refused before anything was assigned: the sequencer runs on as before.
    CHECK(peek(seq, 12) == before);
}

TEST_CASE("load_state refuses a seq_perm entry out of range (manifest)")
{
    // The issue's reproducer: with a manifest the bad entry indexed index_order_
    // before the cursor-mismatch check could run.
    sample_sequencer seq(10, 1, /*shuffle=*/true, /*seed=*/0, {2, 3, 4, 5});
    auto tree = with_perm(seq, {9999999, 1, 2, 3});
    tree.fields["seq_pos"] = std::uint64_t{0};
    const auto before = peek(seq, 12);
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_THROWS_AS(seq.load_state(r), std::runtime_error);
    CHECK(peek(seq, 12) == before);
}

TEST_CASE("load_state refuses a seq_perm with duplicate entries")
{
    for (const bool manifest : {false, true}) {
        CAPTURE(manifest);
        sample_sequencer seq = manifest
            ? sample_sequencer(10, 1, /*shuffle=*/true, /*seed=*/0, {2, 3, 4, 5})
            : sample_sequencer(4, 1, /*shuffle=*/true, /*seed=*/0);
        // In range, right length, but 1 twice and 0 never: not a permutation.
        const auto tree = with_perm(seq, {1, 1, 2, 3});
        const auto before = peek(seq, 12);
        logging::mem_state_reader r(tree, logging::k_format_version);
        CHECK_THROWS_AS(seq.load_state(r), std::runtime_error);
        CHECK(peek(seq, 12) == before);
    }
}

TEST_CASE("a refused manifest-mismatch load leaves the sequencer unchanged")
{
    // The S.5.15 refusal comes after the cursor, RNG and permutation are read;
    // it must not leave them half-adopted.
    sample_sequencer src(20, 1, /*shuffle=*/true, /*seed=*/5, {9, 4, 0, 7, 2});
    run(src, 3);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(20, 1, /*shuffle=*/true, /*seed=*/6, {3, 8, 1, 6, 5});
    run(dst, 2);
    const auto before = peek(dst, 15);
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_THROWS_AS(dst.load_state(r), std::runtime_error);
    CHECK(peek(dst, 15) == before);
}

TEST_CASE("a seq_perm of another length is regenerated (cross-dataset load)")
{
    // Kept: a checkpoint from a larger dataset carries a permutation of the
    // wrong length, which is not corruption -- the sequencer builds a fresh one.
    sample_sequencer src(6, 1, /*shuffle=*/true, /*seed=*/3);
    run(src, 2);
    logging::mem_state_writer w;
    src.save_state(w);
    const auto tree = w.take_root();

    sample_sequencer dst(4, 1, /*shuffle=*/true, /*seed=*/3);
    logging::mem_state_reader r(tree, logging::k_format_version);
    REQUIRE_NOTHROW(dst.load_state(r));
    // pos 2 of 4: the rest of this epoch is two distinct in-range indices, and
    // the next epoch is a full permutation of [0, 4).
    const auto rest = run(dst, 2);
    CHECK(rest[0] < 4);
    CHECK(rest[1] < 4);
    CHECK(rest[0] != rest[1]);
    auto next = run(dst, 4);
    std::ranges::sort(next);
    CHECK(next == std::vector<std::size_t>{0, 1, 2, 3});
}


// SPEC S.5.14: a load restores the sequence iff it continues the run that wrote the
// checkpoint (state_reader::continues_run, set by apply_state from the run ids). The
// checkpoint is a sequencer saved mid-sequence; a load that restores continues from
// there, one that does not keeps the fresh sequencer's start.
TEST_CASE("restore_sequenced_state restores exactly when the load continues the run")
{
    sample_sequencer saved(8, 1, /*shuffle=*/true, /*seed=*/3);
    fast_rng_t saved_rng(5);
    (void)run(saved, 5);
    logging::mem_state_writer w;
    dataset::save_sequenced_state(dataset::sequence_params_t{}, w, saved, saved_rng);
    const auto tree = w.take_root();

    sample_sequencer fresh(8, 1, /*shuffle=*/true, /*seed=*/3);
    const auto fresh_next = peek(fresh, 6);
    const auto continued_next = peek(saved, 6);
    REQUIRE(fresh_next != continued_next);   // the two outcomes are told apart

    for (const bool continues : { true, false }) {
        CAPTURE(continues);
        sample_sequencer seq(8, 1, /*shuffle=*/true, /*seed=*/3);
        fast_rng_t rng(5);
        logging::mem_state_reader r(tree, logging::k_format_version);
        r.set_continues_run(continues);
        CHECK(dataset::restore_sequenced_state(dataset::sequence_params_t{}, r, seq, rng, "test")
              == continues);
        CHECK(peek(seq, 6) == (continues ? continued_next : fresh_next));
    }
}

TEST_CASE("a restoring load checks the dataset's identity record; a fresh one does not read it")
{
    dataset::sequence_params_t saved;
    saved.run.identity = R"({"image_file":"a"})";
    sample_sequencer seq(4, 1, /*shuffle=*/false, /*seed=*/0);
    fast_rng_t rng(1);
    logging::mem_state_writer w;
    dataset::save_sequenced_state(saved, w, seq, rng);
    const auto tree = w.take_root();
    REQUIRE(tree.groups.contains("config"));
    CHECK(tree.groups.at("config").attrs.at("dataset") == saved.run.identity);

    std::vector<std::string> warned;
    coast::diag::set_sink([&](coast::diag::level, std::string_view cat, std::string_view msg) {
        if (cat == "checkpoint") warned.emplace_back(msg);
    });
    // Other data, same run: the load restores and warns, naming both; it does not refuse.
    dataset::sequence_params_t other;
    other.run.identity = R"({"image_file":"b"})";
    logging::mem_state_reader r(tree, logging::k_format_version);
    r.set_continues_run(true);
    CHECK(dataset::restore_sequenced_state(other, r, seq, rng, "ds 7"));
    CHECK(r.take_unrecorded_configs().empty());   // the record was there and was read
    REQUIRE(warned.size() == 1);
    CHECK(warned[0].find("ds 7") != std::string::npos);
    CHECK(warned[0].find(R"(dataset {"image_file":"a"} -> {"image_file":"b"})") != std::string::npos);
    // The same data: silent. Not restoring (another run): the record is not read at all.
    warned.clear();
    logging::mem_state_reader same(tree, logging::k_format_version);
    same.set_continues_run(true);
    CHECK(dataset::restore_sequenced_state(saved, same, seq, rng, "ds 7"));
    logging::mem_state_reader fresh(tree, logging::k_format_version);
    CHECK_FALSE(dataset::restore_sequenced_state(other, fresh, seq, rng, "ds 7"));
    CHECK(warned.empty());
    coast::diag::set_sink({});
}

