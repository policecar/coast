// Death-test helper for B-tier test "observe dtype/shape lock aborts".
//
// The observe dtype/shape lock (observe_registry::resolve_site) is a hard
// invariant: a second OBSERVE against the same site with a different dtype is a
// programming error that std::abort()s — silently writing mismatched data is
// worse than a crash. abort() cannot be caught in-process, so the guard is
// verified by running THIS helper as a subprocess and asserting it terminates
// abnormally (see test_b_tier.cpp).
//
// Contract: if the guard fires we never return (SIGABRT). If we DO return 0,
// the guard failed to fire — the parent test treats a clean exit as failure.

#include "observe/observe_registry.h"

#include <hdf5.h>

#include <cstdio>

int main() {
    auto& reg = observe::observe_registry::instance();

    // Lock the site as a 4-byte float scalar.
    reg.resolve_site(nullptr, "death", "site", H5T_NATIVE_FLOAT, {}, sizeof(float));

    // Re-resolve with a conflicting dtype/size — the lock guard must abort here.
    reg.resolve_site(nullptr, "death", "site", H5T_NATIVE_DOUBLE, {}, sizeof(double));

    // Unreachable if the guard works.
    std::fprintf(stderr, "BUG: observe dtype mismatch did not abort\n");
    return 0;
}
