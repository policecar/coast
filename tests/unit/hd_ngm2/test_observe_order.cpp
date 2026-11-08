// tests/unit/hd_ngm2/test_observe_order.cpp
//
// population/in and get_inp_ids() follow the group's input_channels order,
// however the channels are wired. The recording goes through the real observe
// path (accessor, registry, HDF5 writer) and is read back from the file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5_root_links.h"
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "hdf5/executor.h"
#include "observe/observe_accessor.h"
#include "observe/observe_cfg.h"
#include "observe/observe_registry.h"
#include "observe/observe_trigger.h"
#include "sim_core/io_buffer.h"
#include "sim_core/sim_env.h"

#include <hdf5.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr sim::output_id_t k_group_id = 0;

// Channel A and B have different widths and contents, so a swap changes the
// recorded row.
const std::vector<float> k_a = { 0.10f, 0.20f, 0.30f };
const std::vector<float> k_b = { 0.50f, 0.60f, 0.70f, 0.80f, 0.90f };

// A group whose neurons have one proximal dendrite fed by every channel. With
// `two_channels` the group is fed "A" and "B", otherwise "A" only; it declares
// `group_channels` as its input_channels.
ngm2::cfg_store make_store(std::vector<std::string> group_channels, bool two_channels = true)
{
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels = two_channels ? std::vector<std::string>{ "da", "db" } : std::vector<std::string>{ "da" };
    cs.dendrites["prox"] = dp;

    ngm2::neuron_params_t np;
    np.input_channels   = two_channels ? std::vector<std::string>{ "na", "nb" } : std::vector<std::string>{ "na" };
    np.dendrites        = { { "p", "prox", 2 } };
    np.channel_mappings = { { "na", "p", "da" } };
    if (two_channels) np.channel_mappings.push_back({ "nb", "p", "db" });
    cs.neurons["n"] = np;

    ngm2::neuron_group_params_t gp;
    gp.input_channels   = std::move(group_channels);
    gp.neurons          = { { "pyr", "n", 3 } };
    gp.channel_mappings = { { "A", "pyr", "na" } };
    if (two_channels) gp.channel_mappings.push_back({ "B", "pyr", "nb" });
    cs.neuron_groups["g"] = gp;
    cs.frozen           = true;   // a pure forward pass is enough to record the input
    return cs;
}

using input_map_t = std::unordered_map<std::string, sim::output_channel_t>;

// The message of the runtime_error init_io throws; empty if it succeeds.
std::string init_error(const ngm2::cfg_store& cs, const input_map_t& m)
{
    ngm2::neuron_group_t g(cs, "g", k_group_id);
    try {
        g.init_io(m);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool contains(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

void fill(sim::io_buffer_cpu& buf, const std::vector<float>& v)
{
    std::ranges::copy(v, buf.cur_write_buffer().begin());
    buf.swap_buffer();   // process() reads the read page
}

// The name of the root dataset ending in "__<local_id>" (not a step-index
// sibling); exactly one must exist.
std::string value_dataset(hid_t file, std::string_view local_id)
{
    const std::string suffix = "__" + std::string(local_id);
    std::string found;
    for (const auto& nm : coast::test::root_link_names(file)) {
        if (nm.ends_with(suffix)) {
            REQUIRE(found.empty());
            found = nm;
        }
    }
    return found;
}

// Runs one observe session: each step builds and drives a group and records one
// population/in row. Returns the rows read back from the HDF5 file.
std::vector<std::vector<float>> record_rows(const std::string& dir_name,
                                            const std::vector<std::function<void(observe::observe_accessor&)>>& steps)
{
    const fs::path out_dir = fs::temp_directory_path() / dir_name;
    std::error_code ec;
    fs::remove_all(out_dir, ec);
    {
        auto& reg = observe::observe_registry::instance();
        observe::observe_cfg_t ocfg;
        ocfg.groups["population"] = {};
        hdf5::executor h5;
        // libhdf5 initialises itself on the executor's thread; wait for that before
        // the first observe() names an HDF5 type on this one, as a real run does.
        h5.submit_sync([] {});
        sim::env env;   // only hosts the registry's hook; the test drives the group itself
        reg.attach(env, ocfg, out_dir, h5);
        observe::observe_accessor acc(reg);
        acc.trigger("population", observe::trigger_mode_t::START, true);
        for (const auto& step : steps) step(acc);
        reg.detach();                      // flushes the writer
    }                                  // the executor stops: libhdf5 is free for the read below

    const fs::path h5_path = out_dir / "population.observe.h5";
    REQUIRE(fs::exists(h5_path));
    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const std::string ds_name = value_dataset(file, "in");
    REQUIRE_FALSE(ds_name.empty());
    const hid_t ds = H5Dopen2(file, ("/" + ds_name).c_str(), H5P_DEFAULT);
    REQUIRE(ds >= 0);
    const hid_t sp = H5Dget_space(ds);
    REQUIRE(H5Sget_simple_extent_ndims(sp) == 2);
    hsize_t dims[2] = { 0, 0 };
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    std::vector<float> flat(static_cast<std::size_t>(dims[0] * dims[1]));
    REQUIRE(H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()) >= 0);
    H5Sclose(sp);
    H5Dclose(ds);
    H5Fclose(file);
    fs::remove_all(out_dir, ec);

    std::vector<std::vector<float>> rows;
    for (std::size_t r = 0; r < dims[0]; ++r)
        rows.emplace_back(flat.begin() + r * dims[1], flat.begin() + (r + 1) * dims[1]);
    return rows;
}

std::vector<float> concat(const std::vector<float>& x, const std::vector<float>& y)
{
    std::vector<float> v = x;
    v.insert(v.end(), y.begin(), y.end());
    return v;
}

} // namespace

TEST_CASE("population/in and get_inp_ids follow the declared channel order, whatever the wiring")
{
    // The ids of A and B swapped or not, and the channel map and buffer
    // hand-over A first or B first: an order taken from ids, map or hand-over
    // differs from the declared one in at least one wiring. Both declarations
    // are recorded, so an order sorted by name fails for { B, A }.
    struct wiring { sim::output_id_t id_a, id_b; bool b_first; };
    const std::vector<wiring> wirings = { { 3, 7, false }, { 3, 7, true }, { 7, 3, false }, { 7, 3, true } };
    const std::vector<std::vector<std::string>> declarations = { { "A", "B" }, { "B", "A" } };

    std::vector<std::function<void(observe::observe_accessor&)>> steps;
    std::vector<std::vector<float>> expected;
    for (const auto& decl : declarations) {
        const bool a_first = decl.front() == "A";
        for (const auto& w : wirings) {
            expected.push_back(a_first ? concat(k_a, k_b) : concat(k_b, k_a));
            steps.push_back([decl, w, a_first](observe::observe_accessor& acc) {
                CAPTURE(decl.front());
                CAPTURE(w.id_a);
                CAPTURE(w.b_first);
                const auto cs = make_store(decl);
                ngm2::neuron_group_t g(cs, "g", k_group_id);

                input_map_t m;
                const std::pair<std::string, sim::output_channel_t> a{ "A", { w.id_a, k_a.size() } };
                const std::pair<std::string, sim::output_channel_t> b{ "B", { w.id_b, k_b.size() } };
                if (w.b_first) { m.insert(b); m.insert(a); } else { m.insert(a); m.insert(b); }
                g.init_io(m);

                const auto ids = g.get_inp_ids();
                const std::vector<std::size_t> want = a_first ? std::vector<std::size_t>{ w.id_a, w.id_b }
                                                              : std::vector<std::size_t>{ w.id_b, w.id_a };
                CHECK(std::vector<std::size_t>(ids.begin(), ids.end()) == want);

                sim::io_buffer_cpu buf_a(k_a.size()), buf_b(k_b.size()), out(g.get_outp_size());
                fill(buf_a, k_a);
                fill(buf_b, k_b);
                if (w.b_first) { g.set_input_buffer(w.id_b, buf_b); g.set_input_buffer(w.id_a, buf_a); }
                else           { g.set_input_buffer(w.id_a, buf_a); g.set_input_buffer(w.id_b, buf_b); }
                g.set_output_buffer(out);

                REQUIRE(g.process().has_value());
                g.perform_observations(acc);   // one population/in row
            });
        }
    }

    const auto rows = record_rows("coast_ngm2_observe_order", steps);
    REQUIRE(rows.size() == expected.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        CAPTURE(r);
        CHECK(rows[r] == expected[r]);
    }
}

TEST_CASE("a source feeding two channels appears twice in population/in")
{
    // A and B both come from source 3, so the recorded row holds its data once
    // per channel, as ngm3 records it. Both channels feed the same dendrite.
    const auto cs = make_store({ "A", "B" });

    const auto rows = record_rows("coast_ngm2_observe_shared", { [&cs](observe::observe_accessor& acc) {
        ngm2::neuron_group_t g(cs, "g", k_group_id);
        g.init_io({ { "A", { 3, k_a.size() } }, { "B", { 3, k_a.size() } } });

        const auto ids = g.get_inp_ids();
        CHECK(std::vector<std::size_t>(ids.begin(), ids.end()) == std::vector<std::size_t>{ 3, 3 });

        sim::io_buffer_cpu buf(k_a.size()), out(g.get_outp_size());
        fill(buf, k_a);
        g.set_input_buffer(3, buf);
        g.set_output_buffer(out);
        REQUIRE(g.process().has_value());
        g.perform_observations(acc);
    } });
    REQUIRE(rows.size() == 1);
    CHECK(rows[0] == concat(k_a, k_a));
}

TEST_CASE("the group's input_channels must name exactly the channels it is fed, each once")
{
    const input_map_t both = { { "A", { 3, k_a.size() } }, { "B", { 7, k_b.size() } } };

    CHECK(init_error(make_store({ "A", "B" }), both).empty());
    CHECK(init_error(make_store({ "B", "A" }), both).empty());

    const auto missing = init_error(make_store({ "A" }), both);
    CHECK(contains(missing, "ngm2 neuron group 0: input_channels"));
    CHECK(contains(missing, "does not declare 'B', although a group input feeds it"));

    const auto twice = init_error(make_store({ "A", "B", "A" }), both);
    CHECK(contains(twice, "declares 'A' more than once"));

    const auto unfed = init_error(make_store({ "A", "B", "C" }), both);
    CHECK(contains(unfed, "declares 'C', but no group input feeds it"));

    // Several channels need a declaration: without one their order is unknown.
    const auto undeclared = init_error(make_store({}), both);
    CHECK(contains(undeclared, "does not declare"));
}

TEST_CASE("a group fed by a single channel may leave input_channels empty")
{
    const input_map_t one = { { "A", { 3, k_a.size() } } };
    CHECK(init_error(make_store({}, /*two_channels=*/false), one).empty());
    CHECK(init_error(make_store({ "A" }, /*two_channels=*/false), one).empty());
    CHECK(contains(init_error(make_store({ "B" }, /*two_channels=*/false), one), "declares 'B', but no group input feeds it"));
}
