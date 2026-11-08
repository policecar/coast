// Death-test helper for B-tier test "observe emit() cache-hit dtype lock aborts".
//
// Companion to observe_dtype_mismatch_main.cpp, which trips the dtype/shape lock
// on the resolve_site() (cache-miss) path. This helper trips the SAME lock on the
// observe_registry::emit() CACHE-HIT path — the guard added in this PR.
//
// Why a separate helper (and why a real sim): emit() is gated on the group being
// present in the pass-built name index (group_index_) AND active this step. Both
// are populated only by run_observation_pass(), which fires as the post-process
// hook installed by observe attach(). So — unlike resolve_site(), which a unit
// test drives standalone — emit()'s guard is unreachable without a real observing
// sim. We therefore build a minimal (synthetic-dataset, so no downloaded data)
// ngm2 sim with one configured observe group, step it once so the group becomes
// indexed + active, then drive emit() twice against a stub owner:
//
//   1st emit  (double scalar) -> cache miss -> resolve_site() locks the site WIDE,
//                                site_cache_ now holds it.
//   2nd emit  (float scalar)  -> cache HIT  -> emit()'s enforce_site_lock() sees a
//                                dtype/size mismatch and std::abort()s.
//
// The mismatch is deliberately a NARROWING one (8-byte double locked, 4-byte float
// re-emitted): this is the direction that actually over-reads. The writer commits
// site->value_bytes (the locked 8), not the emit's value_bytes, so without the
// guard H5Dwrite would read 8 bytes from the 4-byte staging buffer — exactly the
// heap over-read the guard fixes. (The reverse, widening, would truncate safely.)
//
// Contract: if the guard fires we never return (SIGABRT). A clean exit 0 means the
// guard failed to fire — the parent test (test_b_tier.cpp) treats that as failure,
// and also asserts the abort carried our "dtype/shape lock violated" diagnostic so
// an unrelated crash cannot masquerade as a pass.

#include "app/build_env.h"
#include "cli/options.h"
#include "core/entity_element.h"
#include "hdf5/executor.h"
#include "observe/observe_registry.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_be32(std::ofstream& os, std::uint32_t v) {
    const char b[4] = {static_cast<char>((v >> 24) & 0xFF), static_cast<char>((v >> 16) & 0xFF),
                       static_cast<char>((v >> 8) & 0xFF),  static_cast<char>(v & 0xFF)};
    os.write(b, 4);
}

// Minimal synthetic MNIST IDX pair (mirrors the writer in test_b_tier.cpp) so the
// helper needs no downloaded dataset.
void write_synthetic_mnist(const fs::path& img_path, const fs::path& lbl_path,
                           std::uint32_t count) {
    constexpr std::uint32_t rows = 28, cols = 28;
    const std::size_t pixels = static_cast<std::size_t>(rows) * cols;
    {
        std::ofstream img(img_path, std::ios::binary);
        write_be32(img, 0x00000803u);
        write_be32(img, count);
        write_be32(img, rows);
        write_be32(img, cols);
        std::vector<char> buf(static_cast<std::size_t>(count) * pixels);
        for (std::uint32_t n = 0; n < count; ++n)
            for (std::size_t p = 0; p < pixels; ++p)
                buf[n * pixels + p] = static_cast<char>((n * 31u + p * 7u) & 0xFFu);
        img.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    {
        std::ofstream lbl(lbl_path, std::ios::binary);
        write_be32(lbl, 0x00000801u);
        write_be32(lbl, count);
        std::vector<char> buf(count);
        for (std::uint32_t n = 0; n < count; ++n) buf[n] = static_cast<char>(n % 10u);
        lbl.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
}

} // namespace

int main() {
    // Scratch area: synthetic dataset + derived config + observe output.
    const fs::path scratch = fs::temp_directory_path() / "coast_observe_emit_death";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch);
    const fs::path img = scratch / "img.idx", lbl = scratch / "lbl.idx";
    write_synthetic_mnist(img, lbl, 64);

    // Derive a config from the repo's ngm2 mnist config, repointed at the synthetic
    // dataset and carrying one observe group that is active from step 0 (so a single
    // pass indexes it and marks it active). Absolute paths -> CWD-independent.
    const fs::path cfg_path = scratch / "cfg.json";
    {
        std::ifstream in(fs::path{COAST_REPO_ROOT} / "configs" / "mnist.json");
        if (!in.good()) { std::fprintf(stderr, "FATAL: cannot read base mnist.json\n"); return 2; }
        nlohmann::json j;
        in >> j;
        for (const char* sub : {"max_interval", "default"}) {
            j["datasets"]["mnist"][sub]["image_file"] = img.string();
            j["datasets"]["mnist"][sub]["label_file"] = lbl.string();
        }
        j["observe"] = {
            {"output_dir", (scratch / "out").string()},
            {"groups", {{"population", {
                {"ranges",  nlohmann::json::array({nlohmann::json::array({0u, 1000000000u})})},
                {"combine", "or"},
            }}}},
        };
        std::ofstream o(cfg_path);
        o << j.dump();
    }

    // Build the sim and step it once so the observe pass indexes "population" and
    // marks it active_this_step. sim_instance's ctor attaches the observe registry.
    hdf5::executor h5;
    coast::cli::options opts;
    opts.config_path = cfg_path.string();
    opts.seed_offset = seed_offset_t{ 0 };
    auto sim = coast::app::build_env(opts, h5);
    if (!sim) { std::fprintf(stderr, "FATAL: build_env returned null\n"); return 2; }

    auto* controller = sim->get_thread_controller();
    if (!controller) { std::fprintf(stderr, "FATAL: no thread controller\n"); return 2; }
    const std::uint64_t target = controller->get_step_count() + 3;
    controller->set_step_limit(target);
    controller->set_mode(sim::sim_mode::unlimited);
    while (controller->get_step_count() < target &&
           controller->get_mode() != sim::sim_mode::paused) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    controller->set_mode(sim::sim_mode::paused);

    // The group is now indexed + active. Drive emit() directly with a stub owner
    // (root_entity() == nullptr -> the site's full_id is the bare local id, no
    // structure registry needed). The mismatch is deliberately a NARROWING one:
    // the first emit locks the site WIDE (8-byte double); the second re-emits the
    // same (owner, group, id) NARROWER (4-byte float) -> cache hit. This is the
    // direction that actually over-reads: without the guard, the writer would be
    // handed a 4-byte value for an 8-byte-locked site and read the locked 8 bytes
    // from the 4-byte staging buffer in H5Dwrite (see observe_writer.cpp — the
    // write size is site->value_bytes, not the emit's value_bytes). The guard sees
    // the dtype/size mismatch on the cache hit and aborts before that happens.
    auto& reg = observe::observe_registry::instance();
    core::entity_element stub;   // concrete, default root_entity_ == nullptr

    const double dval = 2.0;
    reg.emit(stub, "population", "guard_probe", H5T_NATIVE_DOUBLE,
             std::span<const hsize_t>{}, &dval, sizeof(double));

    const float fval = 1.0F;
    reg.emit(stub, "population", "guard_probe", H5T_NATIVE_FLOAT,
             std::span<const hsize_t>{}, &fval, sizeof(float));   // must abort here

    std::fprintf(stderr, "BUG: observe emit() cache-hit dtype mismatch did not abort\n");
    return 0;
}
