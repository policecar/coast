// h5browse open_and_map against REAL writer output — both families
// are produced in-process by the actual COAST writers (observe_writer_t,
// buffer_logger + minimal sim::env), so family detection and
// mapping are tested against the writers' ground truth, not a replica.
//
// In-process integration by tier (like unit.observe.writer_roundtrip): real
// executor worker thread + temp files, no sim binary, sub-second.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/file_model.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"
#include "logging/buffer_logger.h"
#include "observe/observe_site.h"
#include "observe/observe_writer.h"
#include "sim_core/sim_env.h"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path temp_file(const char* name)
{
    const fs::path  p = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove(p, ec);
    return p;
}

const h5b::dataset_desc* find_dataset(const h5b::file_model& m,
                                      std::string_view display)
{
    for (const auto& d : m.datasets) {
        if (d.display_name == display) return &d;
    }
    return nullptr;
}

// Minimal entity so sim::env creates one real io_buffer for buffer_logger.
struct emitting_entity : sim::io_entity {
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override
    {
        return sim::output_id_t{1};
    }
    [[nodiscard]] std::size_t get_outp_size() const override { return 4; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override
    {
        return {};
    }
};

} // namespace

TEST_CASE("open_and_map: stamped observe file (real observe_writer)")
{
    const fs::path path = temp_file("h5b_fm_observe.h5");

    float                  scalar_scratch = 0.0f;
    observe::observe_site_t scalar_site;
    scalar_site.full_id     = "neuron[3]/activity";
    scalar_site.h5_dtype    = H5T_NATIVE_FLOAT;
    scalar_site.shape       = {};
    scalar_site.value_bytes = sizeof(float);

    std::uint32_t           vec_scratch[8] = {};
    observe::observe_site_t vec_site;
    vec_site.full_id     = "population/spike_counts";
    vec_site.h5_dtype    = H5T_NATIVE_UINT32;
    vec_site.shape       = {8};
    vec_site.value_bytes = sizeof(vec_scratch);

    hdf5::executor exec;
    {
        observe::observe_writer_t writer(path, exec);
        writer.start();
        for (std::uint64_t s = 0; s < 5; ++s) {
            scalar_scratch = static_cast<float>(s);
            writer.enqueue(&scalar_site, &scalar_scratch, sizeof(scalar_scratch), s * 10);
            writer.enqueue(&vec_site, vec_scratch, sizeof(vec_scratch), s * 10);
        }
        writer.stop_and_join();
    }

    const auto r = h5b::open_and_map(exec, path);
    REQUIRE(r.has_value());
    const h5b::file_model& m = *r;

    CHECK(m.family == h5b::file_family::observe);
    CHECK(m.schema_version == "1.0");
    CHECK(m.file_bytes > 0);
    REQUIRE(m.datasets.size() == 2);

    const auto* act = find_dataset(m, "neuron[3]/activity");
    REQUIRE(act != nullptr);
    CHECK(act->dtype == h5b::scalar_type::f32);
    CHECK(act->rows == 5);
    CHECK(act->row_width == 1);
    CHECK(act->shape.empty());
    REQUIRE(act->step_index_path.has_value());
    CHECK(*act->step_index_path == "/neuron[3]__activity__step_index");
    CHECK_FALSE(act->compressed); // observe writes uncompressed
    REQUIRE(act->chunk.size() == 1);
    CHECK(act->chunk[0] == 1024); // writer's chunk[0]

    const auto* vec = find_dataset(m, "population/spike_counts");
    REQUIRE(vec != nullptr);
    CHECK(vec->dtype == h5b::scalar_type::u32);
    CHECK(vec->rows == 5);
    CHECK(vec->row_width == 8);
    REQUIRE(vec->shape.size() == 1);
    CHECK(vec->shape[0] == 8);

    // Semantic tree: flat "__" names un-flattened into hierarchy.
    REQUIRE(m.root.children.size() == 2);
    const auto& n3 = m.root.children[0];
    CHECK(n3.name == "neuron[3]");
    CHECK(n3.kind == h5b::tree_node::kind_t::group);
    REQUIRE(n3.children.size() == 1);
    CHECK(n3.children[0].name == "activity");
    CHECK(n3.children[0].kind == h5b::tree_node::kind_t::dataset);
    CHECK(n3.children[0].dataset == 0);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("open_and_map: buffer log (real buffer_logger over minimal env)")
{
    const fs::path path = temp_file("h5b_fm_buffer.h5");
    hdf5::executor exec;
    {
        sim::env env;
        env.emplace_back<emitting_entity>();
        env.init_io_buffers();

        logging::buffer_logger_config cfg;
        cfg.output_path = path.string();
        cfg.chunk_size  = 2; // small: force chunked appends
        // compression stays at the writer default (gzip)

        logging::buffer_logger logger(env, cfg, exec);
        REQUIRE(static_cast<bool>(logger.start()));
        for (int i = 0; i < 5; ++i) {
            REQUIRE(env.process().has_value());
            REQUIRE(env.swap_io().has_value());
        }
        REQUIRE(static_cast<bool>(logger.stop()));
    }

    const auto r = h5b::open_and_map(exec, path);
    REQUIRE(r.has_value());
    const h5b::file_model& m = *r;

    CHECK(m.family == h5b::file_family::buffer_log);
    CHECK(m.schema_version == "1.0");

    const auto* steps = find_dataset(m, "step_numbers");
    REQUIRE(steps != nullptr);
    CHECK(steps->dtype == h5b::scalar_type::u64);
    CHECK(steps->rows == 5);

    // Exactly one buffer dataset (display "buffer 1..." — id from the
    // emitting entity), 2-D [time, 4], gzip'd by default, step-indexed.
    const h5b::dataset_desc* buf = nullptr;
    for (const auto& d : m.datasets) {
        if (d.display_name.starts_with("buffer ")) buf = &d;
    }
    REQUIRE(buf != nullptr);
    CHECK(buf->dtype == h5b::scalar_type::f32);
    CHECK(buf->rows == 5);
    CHECK(buf->row_width == 4);
    CHECK(buf->compressed); // writer default gzip+shuffle
    REQUIRE(buf->step_index_path.has_value());
    CHECK(*buf->step_index_path == "/steps/step_numbers");
    REQUIRE(buf->chunk.size() == 2);
    CHECK(buf->chunk[0] == 2); // cfg.chunk_size

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("open_and_map: simulation state checkpoint is rejected with triage text")
{
    const fs::path path = temp_file("h5b_fm_state.h5");
    hdf5::executor exec;
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        // Real checkpoints stamp type on /metadata (state_logger.cpp), not root.
        const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                    H5P_DEFAULT);
        REQUIRE(md >= 0);
        REQUIRE(hdf5::write_attr(md, "type", std::string_view{"simulation_state"}) >= 0);
        H5Gclose(md);
        H5Fclose(fid);
    });

    const auto r = h5b::open_and_map(exec, path);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().category == h5b::error_category::schema);
    CHECK(r.error().message.find("state checkpoint") != std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("open_and_map: missing file and unrecognized-but-valid HDF5")
{
    hdf5::executor exec;

    const auto missing =
        h5b::open_and_map(exec, "/nonexistent/dir/nothing.h5");
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().category == h5b::error_category::open);

    // Valid HDF5, empty root: recognizable but no COAST family.
    const fs::path path = temp_file("h5b_fm_unknown.h5");
    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        H5Fclose(fid);
    });
    const auto unknown = h5b::open_and_map(exec, path);
    REQUIRE_FALSE(unknown.has_value());
    CHECK(unknown.error().category == h5b::error_category::schema);
    CHECK(unknown.error().message.find("not a recognized COAST log") !=
          std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}
