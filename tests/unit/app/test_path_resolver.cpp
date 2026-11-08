// Unit tests for the per-config output path resolver (coast::paths).
//
// The resolver is the one home for the app-config `config_base` policy: a leaf
// output name (snapshot_dir, observe dir, log dir) is placed under an expanded
// base template, unless the leaf is absolute (then it wins). These tests pin the
// SEMANTICS — including the bare-filename edge case that a naive string
// substitution would silently turn into an absolute filesystem-root path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <filesystem>
#include <string>

#include "core/path_resolver.h"

namespace fs = std::filesystem;
using coast::paths::resolve_config_output;
using coast::paths::leaf_escapes_config_base;

namespace {
// Normalize for a stable, implementation-independent comparison ("./a" -> "a",
// trailing slashes collapsed) while keeping forward slashes.
std::string norm(const fs::path& p) { return p.lexically_normal().generic_string(); }
} // namespace

TEST_CASE("path_resolver: default base places outputs in a per-config subdir next to the config") {
    const auto r = resolve_config_output("configs/mnist.json", "snapshots",
                                         "<config_dir>/<config>/");
    CHECK_FALSE(r.is_absolute());
    CHECK(norm(r) == "configs/mnist/snapshots");
}

TEST_CASE("path_resolver: an absolute leaf wins over the base") {
    const auto r = resolve_config_output("configs/mnist.json", "/tmp/snaps",
                                         "<config_dir>/<config>/");
    CHECK(r.is_absolute());
    CHECK(norm(r) == "/tmp/snaps");
}

TEST_CASE("path_resolver: bare-filename config resolves CWD-relative, NEVER at filesystem root") {
    // The c1 bug: path("mnist.json").parent_path() is empty. A naive
    // "<config_dir>/<config>" string substitution would yield "/mnist/..." —
    // absolute, at the root. Component-wise composition must keep it relative.
    const auto r = resolve_config_output("mnist.json", "snapshots",
                                         "<config_dir>/<config>/");
    CHECK_FALSE(r.is_absolute());
    CHECK(norm(r) == "mnist/snapshots");
}

TEST_CASE("path_resolver: a relative template stays CWD-relative") {
    const auto r = resolve_config_output("configs/mnist.json", "snapshots",
                                         "./runs/<config>/");
    CHECK_FALSE(r.is_absolute());
    CHECK(norm(r) == "runs/mnist/snapshots");
}

TEST_CASE("path_resolver: an absolute template stays absolute") {
    const auto r = resolve_config_output("configs/mnist.json", "snapshots",
                                         "/data/<config>/");
    CHECK(r.is_absolute());
    CHECK(norm(r) == "/data/mnist/snapshots");
}

TEST_CASE("path_resolver: an absolute config path yields an absolute base") {
    const auto r = resolve_config_output("/abs/configs/mnist.json", "snapshots",
                                         "<config_dir>/<config>/");
    CHECK(r.is_absolute());
    CHECK(norm(r) == "/abs/configs/mnist/snapshots");
}

TEST_CASE("path_resolver: a multi-segment relative leaf is placed under the base") {
    // GUI logs use a "logs/buffer_log.h5"-style leaf.
    const auto r = resolve_config_output("configs/mnist.json", "logs/buffer_log.h5",
                                         "<config_dir>/<config>/");
    CHECK_FALSE(r.is_absolute());
    CHECK(norm(r) == "configs/mnist/logs/buffer_log.h5");
}

TEST_CASE("path_resolver: <config> substitutes anywhere within a path component") {
    // Not just as a whole component — mid-segment too, since the stem is a single
    // filename fragment. Guards the fix for the "run_<config> left literal" footgun.
    CHECK(norm(resolve_config_output("configs/mnist.json", "snapshots", "run_<config>/"))
          == "run_mnist/snapshots");
    CHECK(norm(resolve_config_output("configs/mnist.json", "snapshots", "<config>_out/"))
          == "mnist_out/snapshots");
    // Multiple occurrences in one segment all expand.
    CHECK(norm(resolve_config_output("configs/mnist.json", "s", "<config>-<config>/"))
          == "mnist-mnist/s");
}

TEST_CASE("path_resolver: <config_dir> embedded in a larger segment is left literal") {
    // <config_dir> can be multi-segment/absolute, so mid-segment splicing is
    // undefined — it stays literal rather than producing a mangled path. (The
    // resolver is silent; the misuse is warned about once at app-config load.)
    const auto r = resolve_config_output("configs/mnist.json", "snapshots",
                                         "x<config_dir>y/");
    CHECK(norm(r) == "x<config_dir>y/snapshots");
}

TEST_CASE("path_resolver: an empty config_base degenerates to the bare CWD-relative leaf") {
    // An empty template expands to nothing, so the (relative) leaf is left as-is
    // and resolves against the CWD — no per-config subdir. Pins the degenerate
    // case so a future change to expand_config_base can't silently alter it.
    const auto rel = resolve_config_output("configs/mnist.json", "snapshots", "");
    CHECK_FALSE(rel.is_absolute());
    CHECK(norm(rel) == "snapshots");

    // An absolute leaf still wins even with an empty base.
    const auto abs = resolve_config_output("configs/mnist.json", "/tmp/snaps", "");
    CHECK(abs.is_absolute());
    CHECK(norm(abs) == "/tmp/snaps");
}

TEST_CASE("leaf_escapes_config_base: a relative '..' leaf that climbs above the base is rejected") {
    // These would resolve OUTSIDE the config_base tree; the config loaders reject
    // them at load so the pure resolver never has to.
    CHECK(leaf_escapes_config_base(".."));
    CHECK(leaf_escapes_config_base("../snaps"));
    CHECK(leaf_escapes_config_base("../../etc"));
    CHECK(leaf_escapes_config_base("a/../../b"));   // a '..' component anywhere
}

TEST_CASE("leaf_escapes_config_base: ordinary relative leaves are accepted") {
    CHECK_FALSE(leaf_escapes_config_base(""));
    CHECK_FALSE(leaf_escapes_config_base("snapshots"));
    CHECK_FALSE(leaf_escapes_config_base("logs/buffer_log.h5"));
    CHECK_FALSE(leaf_escapes_config_base("./snapshots"));   // '.' is not '..'
    CHECK_FALSE(leaf_escapes_config_base("..hidden"));      // '..' only as a whole component
    CHECK_FALSE(leaf_escapes_config_base("a..b"));
}

TEST_CASE("leaf_escapes_config_base: an absolute leaf is a sanctioned escape, never flagged") {
    // resolve_config_output honours an absolute leaf verbatim; a '..' inside one is
    // just normalization, not an escape of a base it already bypasses.
    CHECK_FALSE(leaf_escapes_config_base("/tmp/snaps"));
    CHECK_FALSE(leaf_escapes_config_base("/tmp/../snaps"));
}
