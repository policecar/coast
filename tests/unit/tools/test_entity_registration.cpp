// tests/unit/tools/test_entity_registration.cpp
//
// Every entity type a config can name must have a create and an init function
// (sim_builder_interface.h). A missing one used to surface only when a config
// used that type, at build_local (#204); this checks every enumerator instead.
// GPU types are registered only in a GPU-enabled build.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "json_cfg/json_loader.h"
#include "json_cfg/sim_json.h"
#include "sim_core/sim_env.h"
#include "tools/sim_builder.h"
#include "tools/sim_builder_interface.h"

TEST_CASE("unit.tools.entity_registration: every entity type has a create and an init function") {
    loader::json_loader jl("configs/mnist.json");   // any config: registration only captures its blocks
    hdf5::executor      h5;
    sim::env            e;
    sim::sim_builder    b(e);
    sim::register_entity_handlers(b, jl, h5);

    using T = sim::io_entity_type_t;
    for (int i = 0; i < static_cast<int>(T::ENTITY_COUNT); ++i) {
        const auto t = static_cast<T>(i);
        const std::string name = nlohmann::ordered_json(t).get<std::string>();
        INFO("entity type ", name);
#ifdef COAST_WITH_GPU
        CHECK(b.handles(t));
#else
        CHECK(b.handles(t) == !sim::is_gpu_entity_type(t));
#endif
    }
}
