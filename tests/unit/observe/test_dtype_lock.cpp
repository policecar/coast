// Unit test 1.2 — observe site dtype/shape locking.
//
// `resolve_site()` is the lazy site-discovery entry point. On first call for a
// given (group, full_id) it allocates the site and *locks* its dtype, shape and
// value size; a later call that agrees returns the same site. (A later call
// that DISAGREES aborts the process — that guard is exercised by the B-tier
// death test 1.3, since an abort cannot be caught in-process.)
//
// This test drives resolve_site() in isolation: `owner == nullptr` is the
// supported "unnamespaced" path (resolve_full_id returns the bare local id with
// no dereference), so no entity/sim is needed.
//
// Singleton hygiene: observe_registry is a process-wide singleton, so each case
// uses a DISTINCT group id to avoid cross-case interference.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_registry.h"
#include "observe/observe_site.h"

#include <hdf5.h>

#include <vector>

TEST_CASE("observe dtype lock: a scalar site locks dtype/shape/bytes on first resolve") {
    auto& reg = observe::observe_registry::instance();

    observe::observe_site_t* s = reg.resolve_site(
        /*owner=*/nullptr, /*group_id=*/"dtl_scalar", /*local_id=*/"act",
        H5T_NATIVE_FLOAT, /*shape=*/{}, /*value_bytes=*/sizeof(float));

    REQUIRE(s != nullptr);
    CHECK(s->full_id == "act");  // nullptr owner -> bare local id, no namespacing
    CHECK(s->h5_dtype == H5T_NATIVE_FLOAT);
    CHECK(s->shape.empty());
    CHECK(s->value_bytes == sizeof(float));

    // Idempotent: the same (group, local, dtype, shape) returns the same site.
    observe::observe_site_t* s2 = reg.resolve_site(
        nullptr, "dtl_scalar", "act", H5T_NATIVE_FLOAT, {}, sizeof(float));
    CHECK(s2 == s);
}

TEST_CASE("observe dtype lock: a vector site records its shape and byte size") {
    auto& reg = observe::observe_registry::instance();

    std::vector<hsize_t> shape{8};
    auto* s = reg.resolve_site(nullptr, "dtl_vec", "vec",
                               H5T_NATIVE_FLOAT, shape, 8 * sizeof(float));

    REQUIRE(s != nullptr);
    REQUIRE(s->shape.size() == 1);
    CHECK(s->shape[0] == 8);
    CHECK(s->value_bytes == 8 * sizeof(float));
}

TEST_CASE("observe dtype lock: distinct local ids create distinct sites in one group") {
    auto& reg = observe::observe_registry::instance();

    auto* a = reg.resolve_site(nullptr, "dtl_multi", "a", H5T_NATIVE_FLOAT, {}, sizeof(float));
    auto* b = reg.resolve_site(nullptr, "dtl_multi", "b", H5T_NATIVE_FLOAT, {}, sizeof(float));

    CHECK(a != nullptr);
    CHECK(b != nullptr);
    CHECK(a != b);
    CHECK(reg.group("dtl_multi") != nullptr);  // group was created on demand
}
