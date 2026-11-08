#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "inhibition/inhibition_cfg.h"
#include "inhibition/random_index.h"
#include "logging/mem_state_io.h"

using inhibition::inhibition_params_t;
using inhibition::inhibition_type_t;
using inhibition::random_index_t;
using inhibition::random_index_variant_t;
using logging::mem_group;
using logging::mem_array2d;
using logging::mem_state_reader;
using logging::mem_state_writer;

namespace {

inhibition_params_t make_params(double reach, double strength,
                                std::uint32_t seed = 0)
{
    inhibition_params_t p;
    p.type        = inhibition_type_t::random_index;
    p.ri_variant  = random_index_variant_t::threshold_linear;
    p.ri_reach    = reach;
    p.ri_strength = strength;
    p.ri_seed     = seed_t{ seed };
    return p;
}

// Inject a known permanence matrix into `ri` through the load path (config-wins,
// so the reach/strength used to derive T come from ri's params, the matrix from
// the tree). random_index_t is non-movable (entity_element base), so load in
// place rather than returning by value.
void inject_matrix(random_index_t& ri, std::vector<float> matrix, std::size_t n)
{
    mem_group tree;
    tree.fields["initialized"] = true;
    tree.fields["n"]           = static_cast<std::uint64_t>(n);
    tree.fields["matrix"]      = mem_array2d{std::move(matrix), n, n};
    mem_state_reader r(tree, logging::k_format_version);
    ri.load_state(r, n);
}

bool all_in_unit_interval(std::span<const float> v)
{
    for (float x : v)
        if (x < 0.0f || x > 1.0f)
            return false;
    return true;
}

} // namespace

TEST_CASE("random_index: transform matches hand computation (exact)")
{
    // reach=0.5, strength=0.5. M (zero diagonal):
    //   row0 [ 0  0.1 0.9 ]   row1 [ 0.2  0  0.8 ]   row2 [ 0.3 0.7  0 ]
    // T = (M<0.5)?0.5:0, diag 0:
    //   row0 [ 0  0.5  0 ]    row1 [ 0.5  0  0 ]      row2 [ 0.5  0  0 ]
    const std::vector<float> M{0.0f, 0.1f, 0.9f,
                               0.2f, 0.0f, 0.8f,
                               0.3f, 0.7f, 0.0f};
    random_index_t ri(make_params(0.5, 0.5));
    inject_matrix(ri, M, 3);

    // raw chosen so one output clamps at the lower bound:
    //   inh = [0.5*0.5, 0.5*1.0, 0.5*1.0] = [0.25, 0.5, 0.5]
    //   out = clamp(raw - inh) = [1.0-0.25, 0.5-0.5, 0.25-0.5] = [0.75, 0, 0]
    const std::vector<float> raw{1.0f, 0.5f, 0.25f};
    std::vector<float> out(3, -1.0f);
    ri.inhibit(raw, out);

    CHECK(out == std::vector<float>{0.75f, 0.0f, 0.0f});
}

TEST_CASE("random_index: generated M and derived T have a zero diagonal")
{
    const std::size_t n = 16;
    random_index_t ri(make_params(0.3, 0.5, /*seed=*/42));
    std::vector<float> raw(n, 0.5f), out(n, 0.0f);
    ri.inhibit(raw, out);                 // forces generation

    REQUIRE(ri.size() == n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(ri.permanence()[i * n + i] == 0.0f);
        CHECK(ri.weights()[i * n + i] == 0.0f);
    }
}

TEST_CASE("random_index: determinism — seed fixes M, different seed changes it")
{
    const std::size_t n = 32;
    std::vector<float> raw(n, 0.5f), out(n, 0.0f);

    random_index_t a(make_params(0.3, 0.5, /*seed=*/7));
    random_index_t b(make_params(0.3, 0.5, /*seed=*/7));
    random_index_t c(make_params(0.3, 0.5, /*seed=*/8));
    a.inhibit(raw, out);
    b.inhibit(raw, out);
    c.inhibit(raw, out);

    const std::vector<float> ma(a.permanence().begin(), a.permanence().end());
    const std::vector<float> mb(b.permanence().begin(), b.permanence().end());
    const std::vector<float> mc(c.permanence().begin(), c.permanence().end());

    CHECK(ma == mb);          // identical seed => identical matrix
    CHECK(ma != mc);          // different seed => different matrix
}

TEST_CASE("random_index: threshold semantics — nonzero fraction ~ reach, value == strength")
{
    const std::size_t n = 96;
    const double reach = 0.3;
    const float  strength = 0.5f;
    random_index_t ri(make_params(reach, strength, /*seed=*/123));
    std::vector<float> raw(n, 0.5f), out(n, 0.0f);
    ri.inhibit(raw, out);

    const auto T = ri.weights();
    std::size_t nonzero = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const float w = T[i * n + j];
            CHECK((w == 0.0f || w == strength));   // every nonzero entry == strength
            if (w != 0.0f) ++nonzero;
        }
    }
    // Off-diagonal entries only; expected fraction ~ reach (binomial, std small
    // relative to count at n=96 -> 9120 off-diagonal cells).
    const double off_diag = static_cast<double>(n) * n - n;
    const double fraction = static_cast<double>(nonzero) / off_diag;
    CHECK(fraction == doctest::Approx(reach).epsilon(0.05));
}

TEST_CASE("random_index: bounds and extremes")
{
    const std::size_t n = 8;

    SUBCASE("reach = 0 -> no inhibition, out == raw")
    {
        random_index_t ri(make_params(0.0, 0.5, /*seed=*/1));
        const std::vector<float> raw{0.1f, 0.9f, 0.5f, 0.25f, 0.75f, 0.0f, 1.0f, 0.33f};
        std::vector<float> out(n, -1.0f);
        ri.inhibit(raw, out);
        CHECK(out == raw);
    }

    SUBCASE("reach = 1, strength = 1 -> outputs driven to zero")
    {
        random_index_t ri(make_params(1.0, 1.0, /*seed=*/1));
        std::vector<float> raw(n, 0.5f), out(n, -1.0f);
        ri.inhibit(raw, out);
        for (float x : out) CHECK(x == 0.0f);     // each neuron inhibited by 7*0.5 = 3.5
    }

    SUBCASE("outputs always within [0,1]")
    {
        random_index_t ri(make_params(0.5, 0.8, /*seed=*/9));
        std::vector<float> raw(n, 1.0f), out(n, -1.0f);
        ri.inhibit(raw, out);
        CHECK(all_in_unit_interval(out));
    }
}

TEST_CASE("random_index: save/load round-trip")
{
    const std::size_t n = 24;
    const std::vector<float> raw(n, 0.6f);

    SUBCASE("save after init -> restored M and outputs are bit-identical")
    {
        random_index_t a(make_params(0.3, 0.5, /*seed=*/55));
        std::vector<float> out_a(n, 0.0f);
        a.inhibit(raw, out_a);                  // generate + produce reference output

        mem_state_writer w;
        a.save_state(w);
        mem_group tree = w.take_root();

        random_index_t b(make_params(0.3, 0.5, /*seed=*/55));
        mem_state_reader r(tree, logging::k_format_version);
        b.load_state(r, n);

        const std::vector<float> ma(a.permanence().begin(), a.permanence().end());
        const std::vector<float> mb(b.permanence().begin(), b.permanence().end());
        CHECK(ma == mb);                        // matrix restored exactly

        std::vector<float> out_b(n, 0.0f);
        b.inhibit(raw, out_b);
        CHECK(out_a == out_b);                  // bit-identical outputs
    }

    SUBCASE("save before init -> load -> inhibit lazily generates")
    {
        random_index_t c(make_params(0.3, 0.5, /*seed=*/55));
        mem_state_writer w;
        c.save_state(w);                        // initialized == false
        mem_group tree = w.take_root();

        random_index_t d(make_params(0.3, 0.5, /*seed=*/55));
        mem_state_reader r(tree, logging::k_format_version);
        d.load_state(r, n);                        // stays lazy

        std::vector<float> out_d(n, 0.0f);
        d.inhibit(raw, out_d);                  // generates from the config seed
        REQUIRE(d.size() == n);
        CHECK(all_in_unit_interval(out_d));

        // Equivalent to a fresh instance that was never saved (same seed/params).
        random_index_t fresh(make_params(0.3, 0.5, /*seed=*/55));
        std::vector<float> out_fresh(n, 0.0f);
        fresh.inhibit(raw, out_fresh);
        CHECK(out_d == out_fresh);
    }
}

// The group's configuration record (M.7.8) is written from params(), so params()
// must name the seed the inhibitor actually runs with, not the configured one.
TEST_CASE("random_index: params() carries a seed drawn from time — the seed of M and of the save")
{
    const std::size_t n = 16;
    const std::vector<float> raw(n, 0.6f);

    auto p = make_params(0.3, 0.5, /*seed=*/7);
    p.ri_seed_from_time = true;
    random_index_t a(p);
    CHECK(a.params().ri_seed != seed_t{ 7 });   // drawn (a 32-bit draw is 7 with p = 2^-32)
    CHECK(a.params().ri_seed_from_time);

    std::vector<float> out(n, 0.0f);
    a.inhibit(raw, out);
    random_index_t same(make_params(0.3, 0.5, std::to_underlying(a.params().ri_seed)));
    same.inhibit(raw, out);
    const std::vector<float> ma(a.permanence().begin(), a.permanence().end());
    const std::vector<float> ms(same.permanence().begin(), same.permanence().end());
    CHECK(ma == ms);                            // M comes from the reported seed

    mem_state_writer w;
    a.save_state(w);
    CHECK(std::get<std::uint64_t>(w.root().fields.at("seed")) == std::to_underlying(a.params().ri_seed));
}

// Resume: M comes from the checkpoint, so params() must name the checkpoint's
// seed — or the next save records a seed M was never generated from.
TEST_CASE("random_index: loading an initialized checkpoint takes its seed with M")
{
    const std::size_t n = 16;
    const std::vector<float> raw(n, 0.6f);

    random_index_t a(make_params(0.3, 0.5, /*seed=*/11));
    std::vector<float> out(n, 0.0f);
    a.inhibit(raw, out);
    mem_state_writer w;
    a.save_state(w);
    const mem_group tree = w.take_root();

    random_index_t b(make_params(0.3, 0.5, /*seed=*/22));
    REQUIRE(b.params().ri_seed == seed_t{ 22 }); // differs from the checkpoint before the load
    mem_state_reader r(tree, logging::k_format_version);
    b.load_state(r, n);
    CHECK(b.params().ri_seed == seed_t{ 11 });

    SUBCASE("a checkpoint from before seeds were 32-bit keeps the low 32 bits of a wider seed")
    {
        // A drawn seed used to be 64 bits; the RNG only ever took its low 32.
        mem_group wide = tree;
        wide.fields["seed"] = (std::uint64_t{ 0xDEADBEEF } << 32) | 11u;
        random_index_t e(make_params(0.3, 0.5, /*seed=*/22));
        mem_state_reader re(wide, logging::k_format_version);
        e.load_state(re, n);
        CHECK(e.params().ri_seed == seed_t{ 11 });
    }

    // The live ri_seed depends on the seed offsets (#190), so a pre-init
    // checkpoint resumed under other offsets must still generate the M the saved
    // run would have: its seed is random state too (M.7.8).
    SUBCASE("a checkpoint saved before init takes its seed too (M is generated from it)")
    {
        random_index_t c(make_params(0.3, 0.5, /*seed=*/11));
        mem_state_writer wc;
        c.save_state(wc);                       // initialized == false
        const mem_group lazy = wc.take_root();

        random_index_t d(make_params(0.3, 0.5, /*seed=*/22));
        mem_state_reader rd(lazy, logging::k_format_version);
        d.load_state(rd, n);
        CHECK(d.params().ri_seed == seed_t{ 11 });

        d.inhibit(raw, out);                    // generates M lazily
        a.inhibit(raw, out);                    // a: M from seed 11
        CHECK(std::equal(d.permanence().begin(), d.permanence().end(),
                         a.permanence().begin(), a.permanence().end()));
    }
}

TEST_CASE("random_index: corrupt state (initialized but bad matrix) regenerates lazily")
{
    const std::size_t n = 12;
    const std::vector<float> raw(n, 0.4f);

    // A fresh instance with these params is the reference for "recovered" output.
    random_index_t ref(make_params(0.3, 0.5, /*seed=*/77));
    std::vector<float> out_ref(n, 0.0f);
    ref.inhibit(raw, out_ref);

    SUBCASE("initialized=true but matrix field absent -> lazy regen")
    {
        mem_group tree;
        tree.fields["initialized"] = true;
        tree.fields["n"]           = static_cast<std::uint64_t>(n);
        // deliberately NO "matrix" field

        random_index_t a(make_params(0.3, 0.5, /*seed=*/77));
        mem_state_reader r(tree, logging::k_format_version);
        a.load_state(r, n);                        // detects empty matrix -> not initialized

        std::vector<float> out_a(n, 0.0f);
        a.inhibit(raw, out_a);                  // regenerates from the config seed
        REQUIRE(a.size() == n);
        CHECK(out_a == out_ref);                // identical to a never-saved instance
    }

    SUBCASE("initialized=true but matrix non-square -> lazy regen")
    {
        mem_group tree;
        tree.fields["initialized"] = true;
        tree.fields["n"]           = static_cast<std::uint64_t>(n);
        tree.fields["matrix"]      = mem_array2d{std::vector<float>(2 * 3, 0.1f), 2, 3};

        random_index_t b(make_params(0.3, 0.5, /*seed=*/77));
        mem_state_reader r(tree, logging::k_format_version);
        b.load_state(r, n);                        // rows != cols -> not initialized

        std::vector<float> out_b(n, 0.0f);
        b.inhibit(raw, out_b);
        REQUIRE(b.size() == n);
        CHECK(out_b == out_ref);
    }
}

TEST_CASE("random_index: degenerate sizes")
{
    SUBCASE("n = 1 -> no possible self-inhibition, out == raw")
    {
        // The single neuron's only connection would be the diagonal, forced to 0.
        random_index_t ri(make_params(1.0, 1.0, /*seed=*/3));  // max reach/strength
        const std::vector<float> raw{0.42f};
        std::vector<float> out{-1.0f};
        ri.inhibit(raw, out);
        REQUIRE(ri.size() == 1);
        CHECK(ri.permanence()[0] == 0.0f);      // sole entry is the (forced-0) diagonal
        CHECK(out == raw);                      // raw - 0 = raw
    }

    SUBCASE("n = 0 -> empty inhibit is a no-op")
    {
        random_index_t ri(make_params(0.3, 0.5, /*seed=*/3));
        std::vector<float> raw, out;            // both empty
        ri.inhibit(raw, out);                   // must not crash or allocate junk
        CHECK(ri.size() == 0);
        CHECK(out.empty());
    }
}

TEST_CASE("random_index: large n stays correct and bounded")
{
    const std::size_t n = 256;                  // 64k-entry matrices
    random_index_t ri(make_params(0.2, 0.6, /*seed=*/101));
    std::vector<float> raw(n, 0.7f), out(n, -1.0f);
    ri.inhibit(raw, out);

    REQUIRE(ri.size() == n);
    CHECK(all_in_unit_interval(out));
    for (std::size_t i = 0; i < n; ++i) {       // zero diagonal preserved at scale
        CHECK(ri.permanence()[i * n + i] == 0.0f);
        CHECK(ri.weights()[i * n + i] == 0.0f);
    }
}

// A checkpoint that does not fit is refused before anything is assigned
// (M.7.18): the inhibitor keeps its seed, matrix and output, and the owning
// group, which loads the inhibitor before it commits, stays unchanged too.
TEST_CASE("random_index: a misfit checkpoint is refused and changes nothing")
{
    const std::size_t n = 10;
    const std::vector<float> raw(n, 0.5f);

    random_index_t saved(make_params(0.3, 0.5, /*seed=*/11));
    std::vector<float> out(n, 0.0f);
    saved.inhibit(raw, out);
    mem_state_writer w;
    saved.save_state(w);
    const mem_group good = w.take_root();

    random_index_t live(make_params(0.3, 0.5, /*seed=*/22));
    std::vector<float> out_before(n, 0.0f);
    live.inhibit(raw, out_before);
    const std::vector<float> m_before(live.permanence().begin(), live.permanence().end());

    auto unchanged = [&] {
        CHECK(live.params().ri_seed == seed_t{ 22 });
        CHECK(live.size() == n);
        CHECK(std::equal(live.permanence().begin(), live.permanence().end(),
                         m_before.begin(), m_before.end()));
        std::vector<float> out_after(n, 0.0f);
        live.inhibit(raw, out_after);
        CHECK(out_after == out_before);
    };

    SUBCASE("matrix of another group size")
    {
        mem_state_reader r(good, logging::k_format_version);
        CHECK_THROWS_WITH(live.load_state(r, n + 1),
                          doctest::Contains("the group has 11 neurons"));
        unchanged();
    }

    SUBCASE("n field disagrees with the matrix shape")
    {
        mem_group bad = good;
        bad.fields["n"] = static_cast<std::uint64_t>(n + 3);
        mem_state_reader r(bad, logging::k_format_version);
        CHECK_THROWS_WITH(live.load_state(r, n),
                          doctest::Contains("stored n disagrees with matrix shape"));
        unchanged();
    }

    SUBCASE("the same checkpoint loads into a group of its size")
    {
        mem_state_reader r(good, logging::k_format_version);
        live.load_state(r, n);
        CHECK(live.params().ri_seed == seed_t{ 11 });
        CHECK(std::equal(live.permanence().begin(), live.permanence().end(),
                         saved.permanence().begin(), saved.permanence().end()));
    }
}
