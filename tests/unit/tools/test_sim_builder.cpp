#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "sim_core/sim_env.h"
#include "tools/sim_builder.h"

// Issue #199: a repeated node id used to overwrite the earlier node, so its
// entity vanished from the sim while cfg.sim.nodes still listed it. The id check
// runs while the node map is assembled, before any entity is created, so no
// create functions need to be registered for these configs.

namespace {

sim::sim_node_t node(sim::sim_node_id_t id, std::string label) {
    return { id, std::move(label), sim::io_entity_type_t::mnist_input, "cfg", {}, {} };
}

std::string build_error(const sim::sim_cfg_t& cfg) {
    sim::env e;
    sim::sim_builder b(e);
    try { b.build_local(cfg); }
    catch (const std::exception& ex) { return ex.what(); }
    return "";
}

} // namespace

TEST_CASE("unit.tools.sim_builder: two direct nodes with one id are refused") {
    sim::sim_cfg_t cfg;
    cfg.nodes = { node(3, "first"), node(3, "second") };
    CHECK(build_error(cfg) == "duplicate node id 3: \"first\" already uses it");
}

TEST_CASE("unit.tools.sim_builder: a node_set expanding onto an existing id is refused") {
    sim::sim_cfg_t cfg;
    cfg.nodes     = { node(1, "tmpl"), node(11, "direct") };
    cfg.node_sets = { { 10, "set", { 1 }, {} } };   // 10 + 1 == 11
    CHECK(build_error(cfg) == "duplicate node id 11: \"direct\" already uses it");
}

TEST_CASE("unit.tools.sim_builder: distinct ids pass the id check") {
    // Gets past assembly and fails later, at entity creation, for want of a
    // create function: that error proves the id check let the config through.
    sim::sim_cfg_t cfg;
    cfg.nodes     = { node(1, "tmpl"), node(2, "other") };
    cfg.node_sets = { { 10, "set", { 1 }, {} } };
    CHECK(build_error(cfg).starts_with("No create function"));
}
