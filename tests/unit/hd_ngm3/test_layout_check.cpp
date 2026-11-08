// tests/unit/hd_ngm3/test_layout_check.cpp
//
// ngm3::check_bank_layout and the CPU group's load_state around it.
//
// The compute relies on the replica layout (hd_ngm3_layout.h, L2-L5): the
// response reuses row r-1's penalty draw when the segment id repeats, and the
// split rewrites a whole run from its first row. Learning keeps the layout, so
// only a loaded state can break it; load_state must refuse such a state before
// anything is assigned, or a stale draw is applied (broken coherence) or the
// split writes into the next dendrite (a misaligned run).
//
// The helper is checked exhaustively per column: every reachable column (a full
// binary tree of depth <= L, L = 0..3) is accepted, and every single-row edit
// of one is rejected unless it cannot be observed (a float edit of a run of one
// row). The load cases run a real group to a branched state, so the edited
// checkpoints are ones the compute could otherwise step.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_layout.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "logging/mem_state_io.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- helper: one (neuron, dendrite, input) column ----------------------------

using column_t = std::vector<std::uint8_t>;   // segment id per row

// Every column the layout allows at depth L: the leaf sets of the full binary
// trees of depth <= L, each leaf s written over its rows I(s).
std::vector<column_t> all_columns(unsigned depth)
{
    const std::size_t rows = std::size_t{1} << depth;
    // Columns of the subtree rooted at s, as (row, id) fills over I(s).
    auto expand = [&](auto &self, unsigned s) -> std::vector<column_t> {
        const unsigned    level = static_cast<unsigned>(std::bit_width(s)) - 1u;
        const std::size_t len   = std::size_t{1} << (depth - level);
        const std::size_t begin = (std::size_t{s} << (depth - level)) - rows;
        column_t leaf(rows, 0);
        std::fill_n(leaf.begin() + static_cast<std::ptrdiff_t>(begin), len,
                    static_cast<std::uint8_t>(s));
        std::vector<column_t> out{ leaf };
        if (level == depth) return out;
        for (const auto &a : self(self, 2 * s))
            for (const auto &b : self(self, 2 * s + 1)) {
                column_t c(rows, 0);
                for (std::size_t r = 0; r < rows; ++r) c[r] = a[r] | b[r];
                out.push_back(c);
            }
        return out;
    };
    return expand(expand, 1u);
}

struct bank_t {
    std::vector<float>        perm, adapt, mm;
    std::vector<std::uint8_t> seg;
};

// One column (neuron_cnt = d_cnt = input_size = 1), each run's floats a
// function of its segment id, so replicas are coherent.
bank_t bank_of(const column_t &c)
{
    bank_t b;
    b.seg = c;
    for (const auto s : c) {
        b.perm .push_back(0.01f * static_cast<float>(s));
        b.adapt.push_back(0.02f * static_cast<float>(s));
        b.mm   .push_back(0.03f * static_cast<float>(s));
    }
    return b;
}

bool accepts(const bank_t &b, unsigned depth, std::string *why = nullptr)
{
    return ngm3::check_bank_layout(b.perm, b.seg, b.adapt, b.mm, 1, 1,
                                   static_cast<std::uint8_t>(depth), 1, why);
}

// Rows of the run row r belongs to: 2^(L - level), and c.size() = 2^L.
std::size_t run_length(const column_t &c, std::size_t r)
{
    return c.size() >> (std::bit_width(unsigned{c[r]}) - 1);
}

// ---- load: a small CPU group driven to a branched state ----------------------

struct test_group_t : ngm3::neuron_group_t {
    using ngm3::neuron_group_t::neuron_group_t;
    using ngm3::neuron_group_t::init_storage;
    using ngm3::neuron_group_t::compute_step;
};

constexpr std::size_t   k_input_size = 16;
constexpr std::size_t   k_neuron_cnt = 4;
constexpr std::size_t   k_d_cnt      = 2;
constexpr std::uint8_t  k_depth      = 2;
constexpr std::size_t   k_rows       = std::size_t{1} << k_depth;

const ngm3::cfg_store &store()
{
    static const ngm3::cfg_store cs = [] {
        ngm3::cfg_store s;
        auto &g = s.neuron_groups["g"];
        g.neuron_cnt              = k_neuron_cnt;
        g.apical_cnt              = 0;
        g.proximal_cnt            = k_d_cnt;
        g.max_branch_level   = k_depth;
        g.random_seed             = seed_t{ 7 };
        // Branch early and often, so the checkpoint holds runs of every length.
        g.branch_interval         = 10;
        // Low enough that mismatch alone decides the split. At 0.05 the few
        // qualifying synapses sat at 1.0-1.5x the threshold, so the learning
        // rule or the standard library's distributions (libc++ vs libstdc++)
        // decided whether the group branched at all within 200 steps.
        g.accumulated_theta_thres = 0.01f;
        g.min_mismatch_deviation  = 0.0f;
        g.mismatch_act_thres      = 0.3f;
        // Mismatch is a slow average (default 0.001): at this input width its
        // spread would need thousands of steps to clear the 1/input_size margin.
        g.mismatch_smoothing      = 0.05f;
        return s;
    }();
    return cs;
}

test_group_t make_group()
{
    test_group_t g(store(), "g", /*output_id=*/1);
    g.init_storage(/*apical_in_size=*/0, /*proximal_in_size=*/k_input_size);
    return g;
}

// Two alternating patterns on most inputs, which the winning dendrite learns
// (low mismatch), and seeded noise on the last few, which it cannot (high
// mismatch). That spread is what lets synapses clear the split threshold
// mm_avg + 1/input_size; a uniform drive leaves them all alike and nothing
// branches.
void drive(test_group_t &g, std::size_t steps)
{
    constexpr std::size_t k_noisy = 4;
    std::mt19937 gen(11);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> in(k_input_size);
    for (std::size_t s = 0; s < steps; ++s) {
        for (std::size_t i = 0; i < k_input_size; ++i)
            in[i] = i >= k_input_size - k_noisy ? u(gen)
                  : ((i + s) % 2 == 0)          ? 0.95f : 0.1f;
        sim::throw_on_step_fault(g.compute_step({}, in), "ngm3 compute_step");
    }
}

logging::mem_group state_of(const test_group_t &g)
{
    logging::mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

template <typename T>
std::vector<T> &field(logging::mem_group &tree, const char *name)
{
    return std::get<std::vector<T>>(tree.groups.at("proximal").fields.at(name));
}

std::size_t at(std::size_t n, std::size_t d, std::size_t r, std::size_t k)
{
    return ((n * k_d_cnt + d) * k_rows + r) * k_input_size + k;
}

// load_state must throw, name the law, and leave the group as it was.
void check_refused(logging::mem_group tree, const char *law)
{
    auto g = make_group();
    drive(g, 3);
    const auto before = state_of(g);
    logging::mem_state_reader r(tree, logging::k_format_version);
    std::string what;
    try { g.load_state(r); } catch (const std::runtime_error &e) { what = e.what(); }
    INFO("message: ", what);
    CHECK(what.find("corrupt checkpoint") != std::string::npos);
    CHECK(what.find(law) != std::string::npos);
    CHECK(state_of(g) == before);
}

// A driven group's checkpoint, required to hold a split.
logging::mem_group branched_tree()
{
    auto g = make_group();
    drive(g, 200);
    auto tree = state_of(g);
    const auto &seg = field<std::uint8_t>(tree, "segment_ids");
    REQUIRE_MESSAGE(std::any_of(seg.begin(), seg.end(), [](auto s) { return s > 1; }),
                    "fixture must branch, or the edits below miss split runs");
    return tree;
}

} // namespace

TEST_CASE("every reachable column passes the layout check")
{
    // Full binary trees of depth <= L: 1, 2, 5, 26 (a(L) = a(L-1)^2 + 1).
    const std::size_t expected[] = { 1, 2, 5, 26 };
    for (unsigned depth = 0; depth <= 3; ++depth) {
        const auto cols = all_columns(depth);
        CHECK(cols.size() == expected[depth]);
        for (const auto &c : cols) {
            std::string why;
            CHECK_MESSAGE(accepts(bank_of(c), depth, &why), why);
        }
    }
}

TEST_CASE("every single-row segment-id edit is rejected")
{
    // Each edit breaks range (L2), alignment (L3) or a full run (L4): a row can
    // only hold another id by misplacing it or by leaving its own run short.
    for (unsigned depth = 0; depth <= 3; ++depth) {
        const unsigned max_seg = (2u << depth) - 1u;
        for (const auto &c : all_columns(depth))
            for (std::size_t r = 0; r < c.size(); ++r)
                for (unsigned v = 0; v <= max_seg + 1; ++v) {
                    if (v == c[r]) continue;
                    auto b = bank_of(c);
                    b.seg[r] = static_cast<std::uint8_t>(v);
                    CHECK_FALSE_MESSAGE(accepts(b, depth),
                                        "depth " << depth << " row " << r << " -> " << v);
                }
    }
}

TEST_CASE("a replica row that differs from its run is rejected")
{
    // Bitwise: one ulp in any of the three fields. A run of one row has no
    // replica to differ from, so the same edit there is a valid state.
    for (unsigned depth = 0; depth <= 3; ++depth)
        for (const auto &c : all_columns(depth))
            for (std::size_t r = 0; r < c.size(); ++r)
                for (int f = 0; f < 3; ++f) {
                    auto b = bank_of(c);
                    auto &v = (f == 0 ? b.perm : f == 1 ? b.adapt : b.mm)[r];
                    v = std::nextafter(v, 1.0f);
                    std::string why;
                    const bool ok = accepts(b, depth, &why);
                    if (run_length(c, r) == 1) CHECK(ok);
                    else CHECK_MESSAGE((!ok && why.find("(L5)") != std::string::npos),
                                       "depth " << depth << " row " << r << " field " << f);
                }
}

TEST_CASE("a value the compute cannot hold is rejected, even copied across its run (L1)")
{
    // depth 2, one column, all four rows the root segment: a run of four rows,
    // so a value written into every row passes the bitwise replica compare (L5).
    const column_t root(4, 1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    struct edit_t { int field; float value; bool ok; };
    const edit_t edits[] = {
        { 0, nan,   false }, { 0, 1.5f,  false }, { 0, -0.1f, false },
        { 0, 0.0f,  true  }, { 0, 1.0f,  true  },
        { 1, nan,   false }, { 1, inf,   false }, { 1, -0.5f, false }, { 1, 0.0f, true },
        { 2, nan,   false }, { 2, inf,   false },
        { 2, -0.25f, true },   // mismatch's sign rests on the unvalidated smoothing
    };
    for (const auto &e : edits) {
        auto b = bank_of(root);
        auto &f = (e.field == 0 ? b.perm : e.field == 1 ? b.adapt : b.mm);
        std::fill(f.begin(), f.end(), e.value);
        std::string why;
        const bool ok = accepts(b, 2, &why);
        CHECK_MESSAGE(ok == e.ok, "field " << e.field << " value " << e.value << ": " << why);
        if (!e.ok) CHECK(why.find("(L1)") != std::string::npos);
    }
}

TEST_CASE("the check covers every column and names the one that fails")
{
    const std::size_t n = 2, d = 3, rows = 4, k = 5;
    bank_t b;
    b.perm .assign(n * d * rows * k, 0.5f);
    b.adapt.assign(b.perm.size(), 0.0f);
    b.mm   .assign(b.perm.size(), 0.0f);
    b.seg  .assign(b.perm.size(), 1);
    CHECK(ngm3::check_bank_layout(b.perm, b.seg, b.adapt, b.mm, n, d, 2, k, nullptr));

    // The last cell of the bank: neuron 1, dendrite 2, row 3, input 4.
    b.perm.back() = 0.25f;
    std::string why;
    CHECK_FALSE(ngm3::check_bank_layout(b.perm, b.seg, b.adapt, b.mm, n, d, 2, k, &why));
    CHECK(why.find("neuron 1 dendrite 2 input 4") != std::string::npos);

    // L1 reaches the last cell too: a NaN copied across that cell's whole run, so the
    // replica compare (L5) sees equal rows and only L1 can refuse it.
    b.perm.back() = 0.5f;
    for (std::size_t r = 0; r < rows; ++r)
        b.mm[b.mm.size() - 1 - r * k] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(ngm3::check_bank_layout(b.perm, b.seg, b.adapt, b.mm, n, d, 2, k, &why));
    CHECK(why.find("neuron 1 dendrite 2 input 4") != std::string::npos);
    CHECK(why.find("(L1)") != std::string::npos);

    // Sizes that do not match the shape are rejected, not read past.
    b.perm.pop_back();
    CHECK_FALSE(ngm3::check_bank_layout(b.perm, b.seg, b.adapt, b.mm, n, d, 2, k, &why));
}

TEST_CASE("load_state accepts a checkpoint the group wrote")
{
    const auto tree = branched_tree();
    auto g = make_group();
    logging::mem_state_reader r(tree, logging::k_format_version);
    CHECK_NOTHROW(g.load_state(r));
    CHECK(state_of(g) == tree);
}

TEST_CASE("load_state refuses a replica row edited apart from its run")
{
    auto tree = branched_tree();
    const auto &seg = field<std::uint8_t>(tree, "segment_ids");
    // Find a run of at least two rows: row 1 of a column whose row 0 holds a
    // segment of level < L (rows 0 and 1 then belong to the same run).
    std::size_t idx = 0;
    bool found = false;
    for (std::size_t n = 0; n < k_neuron_cnt && !found; ++n)
        for (std::size_t d = 0; d < k_d_cnt && !found; ++d)
            for (std::size_t k = 0; k < k_input_size && !found; ++k)
                if (std::bit_width(unsigned{seg[at(n, d, 0, k)]}) - 1 < k_depth) {
                    idx = at(n, d, 1, k);
                    found = true;
                }
    REQUIRE(found);
    for (const char *name : { "permanences", "mismatch", "adapt_history" }) {
        auto edited = tree;
        auto &v = field<float>(edited, name)[idx];
        v = std::nextafter(v, 2.0f);
        INFO("field: ", name);
        check_refused(edited, "(L5)");
    }
}

TEST_CASE("load_state refuses a misaligned segment run")
{
    // The shape the audit drove into a cross-dendrite write: a run that does not
    // start at its segment's rows. Swapping the two children of a split column
    // keeps every id in range and every run full; only the alignment breaks.
    auto tree = branched_tree();
    auto &seg = field<std::uint8_t>(tree, "segment_ids");
    std::size_t col = 0;
    bool found = false;
    for (std::size_t c = 0; c < k_neuron_cnt * k_d_cnt * k_input_size && !found; ++c) {
        const std::size_t nd = c / k_input_size, k = c % k_input_size;
        if (seg[at(nd / k_d_cnt, nd % k_d_cnt, 0, k)] == 2 &&
            seg[at(nd / k_d_cnt, nd % k_d_cnt, k_rows - 1, k)] == 3) {
            col = c;
            found = true;
        }
    }
    REQUIRE_MESSAGE(found, "fixture must hold a column split once at the root");
    const std::size_t nd = col / k_input_size, k = col % k_input_size;
    for (std::size_t r = 0; r < k_rows; ++r) {
        auto &s = seg[at(nd / k_d_cnt, nd % k_d_cnt, r, k)];
        if (s == 2) s = 3; else if (s == 3) s = 2;
    }
    check_refused(tree, "(L3)");
}

TEST_CASE("load_state refuses an out-of-range segment id")
{
    auto tree = branched_tree();
    auto &seg = field<std::uint8_t>(tree, "segment_ids");
    std::fill_n(seg.begin(), k_rows * k_input_size, std::uint8_t{0});   // neuron 0 dendrite 0
    check_refused(tree, "(L2)");
}

TEST_CASE("load_state refuses an incomplete or missing per-neuron rng stream")
{
    // The streams are read after the banks: they must be staged with them, or a
    // corrupt stream would throw with the banks already assigned (#166).
    const auto tree = branched_tree();
    REQUIRE(tree.groups.at("neuron_rgens").groups.contains("1"));
    {
        auto edited = tree;
        std::get<std::vector<std::uint32_t>>(
            edited.groups.at("neuron_rgens").groups.at("1").fields.at("mt")).pop_back();
        check_refused(edited, "neuron 1: corrupt checkpoint — rng 1: mt has");
    }
    {
        auto edited = tree;
        edited.groups.at("neuron_rgens").groups.erase("1");
        check_refused(edited, "rng of neuron 1 missing");
    }
}

TEST_CASE("load_state refuses a misfit inhibition part before anything is assigned")
{
    // The inhibitor is loaded before the group commits and is all or nothing
    // itself (inhibitor_t::load_state). It used to be loaded after the commit,
    // so a misfit inhibition part left the banks of the checkpoint assigned.
    auto tree = branched_tree();   // banks differ from a fresh group's
    const std::size_t wrong = k_neuron_cnt + 1;
    auto &inh = tree.groups["inhibition"];
    inh.fields["initialized"] = true;
    inh.fields["n"]           = static_cast<std::uint64_t>(wrong);
    inh.fields["matrix"]      = logging::mem_array2d{ std::vector<float>(wrong * wrong, 0.5f),
                                                      wrong, wrong };

    ngm3::cfg_store cs = store();
    auto &ri       = cs.inhibitions["ri"];
    ri.type        = inhibition::inhibition_type_t::random_index;
    ri.ri_reach    = 0.3;
    ri.ri_strength = 0.5;
    cs.neuron_groups["g"].inhibition_cfg = "ri";
    test_group_t g(cs, "g", /*output_id=*/1);
    g.init_storage(/*apical_in_size=*/0, /*proximal_in_size=*/k_input_size);

    const auto before = state_of(g);
    logging::mem_state_reader r(tree, logging::k_format_version);
    std::string what;
    try { g.load_state(r); } catch (const std::runtime_error &e) { what = e.what(); }
    INFO("message: ", what);
    CHECK(what.find("the group has 4 neurons") != std::string::npos);
    CHECK(state_of(g) == before);
}
