// tests/integration_ngm3/ngm25_xdev_checkpoint_test.cpp
//
// Review should-fix #5: cross-device ngm2.5 checkpoint round-trip.
//
// The CPU (ngm25::neuron_group_t) and GPU (ngm25::neuron_group_gpu_t) save_state /
// load_state are two hand-mirrored, independent implementations that agree today
// only by construction — verified once by line-by-line review, but nothing in CI
// stops a field added to one side (or a rename) from silently desyncing the two.
// State is routed by entity id, not entity_type (state_logger::load_state), so a
// checkpoint written by one device loads straight into the other.
//
// This test closes that gap. In each direction it: builds a source-device sim,
// runs it (past a branch_interval so the arena has grown/repacked), saves; builds
// a fresh destination-device sim, loads that checkpoint, and IMMEDIATELY re-saves
// with zero steps in between. The destination's re-save must reproduce the source
// entity subtree byte-for-byte: any field the loader ignored would come back as
// the destination's fresh-construction default, and any field one side omits shows
// up as a structural difference. We compare the whole /entities/<id> subtree
// (every dataset, both directions) so a new field can't slip through un-checked.
//
// GPU-labeled: it builds a real GPU sim, so it needs an adapter and is excluded
// from CPU-only / sanitizer runs (-LE gpu). GPU ngm2.5 is run-to-run deterministic
// and the arena round-trips through host-visible SoA, so exact comparison is valid.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <hdf5.h>

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "logging/state_logger.h"
#include "sim_core/sim_instance.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t k_steps = 6000;   // > branch_interval (5000): forces a repack
constexpr int           k_group_id = 1;   // the ngm25 group (id 0 is mnist_input)

// Save `path`, running the source sim for k_steps first.
void run_and_save(const char* config, const fs::path& out) {
    hdf5::executor h5;
    coast::cli::options opts;
    opts.config_path = config;
    opts.seed_offset = seed_offset_t{ 0 };
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    coast::test::run_for(*sim, k_steps);

    logging::state_logger sl;
    logging::state_save_config cfg;
    cfg.output_path = out;
    REQUIRE(sl.save_state(sim->get_sim_env(),
                          sim->get_thread_controller()->get_step_count(), cfg, h5).success);
}

// Build a fresh `config` sim, load `in`, and immediately re-save to `out` with
// ZERO steps in between — `out` is the loaded state serialized by `config`'s device.
void load_and_resave(const char* config, const fs::path& in, const fs::path& out) {
    hdf5::executor h5;
    coast::cli::options opts;
    opts.config_path = config;
    opts.seed_offset = seed_offset_t{ 0 };
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    logging::state_logger sl;
    auto lr = sl.load_state(sim->get_sim_env(), in, h5);
    REQUIRE(lr.success);

    logging::state_save_config cfg;
    cfg.output_path = out;
    REQUIRE(sl.save_state(sim->get_sim_env(), lr.restored_step, cfg, h5).success);
}

// --- HDF5 subtree comparison (self-contained; no h5diff dependency) ----------

// Collect the paths of every dataset under `root` (relative to root).
herr_t collect_dataset(hid_t, const char* name, const H5O_info2_t* info, void* op) {
    if (info->type == H5O_TYPE_DATASET)
        static_cast<std::set<std::string>*>(op)->emplace(name);
    return 0;
}

std::set<std::string> dataset_paths(hid_t file, const std::string& root) {
    std::set<std::string> paths;
    H5Ovisit_by_name3(file, root.c_str(), H5_INDEX_NAME, H5_ITER_NATIVE,
                      collect_dataset, &paths, H5O_INFO_BASIC, H5P_DEFAULT);
    return paths;
}

// Read a dataset's raw bytes in its own file datatype (npoints * type_size).
std::vector<char> read_raw(hid_t file, const std::string& path) {
    const hid_t d = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    REQUIRE(d >= 0);
    const hid_t t  = H5Dget_type(d);
    const hid_t sp = H5Dget_space(d);
    const hssize_t n = H5Sget_simple_extent_npoints(sp);
    const size_t sz  = H5Tget_size(t);
    std::vector<char> buf(static_cast<size_t>(n) * sz);
    if (!buf.empty())
        REQUIRE(H5Dread(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) >= 0);
    H5Sclose(sp); H5Tclose(t); H5Dclose(d);
    return buf;
}

// Assert the two files' /entities/<k_group_id> subtrees are byte-identical.
void expect_entity_subtree_identical(const fs::path& a, const fs::path& b) {
    const hid_t fa = H5Fopen(a.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hid_t fb = H5Fopen(b.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(fa >= 0);
    REQUIRE(fb >= 0);

    const std::string root = "/entities/" + std::to_string(k_group_id);
    const auto pa = dataset_paths(fa, root);
    const auto pb = dataset_paths(fb, root);
    CHECK_MESSAGE(pa == pb, "entity subtree has a different SET of datasets across devices "
                            "(a field saved on one side but not the other)");

    for (const auto& rel : pa) {
        if (pb.find(rel) == pb.end()) continue;   // set-diff already reported above
        const std::string full = root + "/" + rel;
        const auto ba = read_raw(fa, full);
        const auto bb = read_raw(fb, full);
        CHECK_MESSAGE(ba.size() == bb.size(), "dataset size differs: " << full);
        if (ba.size() == bb.size())
            CHECK_MESSAGE(std::memcmp(ba.data(), bb.data(), ba.size()) == 0,
                          "dataset bytes differ across devices: " << full);
    }

    H5Fclose(fa); H5Fclose(fb);
}

fs::path scratch_dir() {
    const fs::path d = fs::temp_directory_path() / "coast_ngm25_xdev";
    std::error_code ec;
    fs::create_directories(d, ec);
    return d;
}

} // namespace

// CPU writes the checkpoint, GPU loads + re-serializes it. Byte-equality proves the
// GPU loader consumed every field the CPU saver wrote and the GPU saver re-emitted
// them identically.
TEST_CASE("ngm25 cross-device checkpoint: CPU-save -> GPU-load -> GPU-save round-trips") {
    const fs::path dir = scratch_dir();
    const fs::path cpu = dir / "cpu.h5", gpu = dir / "gpu_reloaded.h5";
    run_and_save(COAST_NGM25_CPU_CONFIG, cpu);
    load_and_resave(COAST_NGM25_GPU_CONFIG, cpu, gpu);
    expect_entity_subtree_identical(cpu, gpu);
}

// The reverse direction: GPU writes, CPU loads + re-serializes.
TEST_CASE("ngm25 cross-device checkpoint: GPU-save -> CPU-load -> CPU-save round-trips") {
    const fs::path dir = scratch_dir();
    const fs::path gpu = dir / "gpu.h5", cpu = dir / "cpu_reloaded.h5";
    run_and_save(COAST_NGM25_GPU_CONFIG, gpu);
    load_and_resave(COAST_NGM25_CPU_CONFIG, gpu, cpu);
    expect_entity_subtree_identical(gpu, cpu);
}
