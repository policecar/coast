#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// End-to-end seam test for the index_manifest plumbing: a *real* loader
// (mnist_io) must read the manifest file path from its config, resolve it
// against the source dataset, and present samples in the manifest's order.
// The sequencer's index_order mapping and the manifest parser are unit-tested
// in isolation in test_sample_sequencer.cpp; this test covers the one wiring
// line each index-based loader shares -- build_sequencer(params, image_count)
// loading params.index_manifest -- using mnist as the witness. The loader is
// driven through its public surface (set_output_buffer + process), matching
// how the sim thread advances it.

#include "io/mnist_io.h"
#include "io/mnist_cfg.h"
#include "sim_core/io_buffer.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_be32(std::ostream& os, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>((v >> 24) & 0xFFu),
        static_cast<unsigned char>((v >> 16) & 0xFFu),
        static_cast<unsigned char>((v >> 8)  & 0xFFu),
        static_cast<unsigned char>( v        & 0xFFu),
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

// Emit a synthetic MNIST-format IDX pair of `count` 28x28 images. Pixel/label
// content is deterministic but irrelevant here -- the test only inspects the
// *order* of presented source indices, not pixel values.
void write_synthetic_mnist(const fs::path& img, const fs::path& lbl,
                           std::uint32_t count) {
    constexpr std::uint32_t rows = 28, cols = 28;
    const std::size_t pixels = static_cast<std::size_t>(rows) * cols;
    {
        std::ofstream o(img, std::ios::binary);
        write_be32(o, 0x00000803u);   // magic: unsigned byte, 3 dims
        write_be32(o, count);
        write_be32(o, rows);
        write_be32(o, cols);
        std::vector<char> buf(static_cast<std::size_t>(count) * pixels);
        for (std::uint32_t n = 0; n < count; ++n)
            for (std::size_t p = 0; p < pixels; ++p)
                buf[n * pixels + p] = static_cast<char>((n * 31u + p * 7u) & 0xFFu);
        o.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
    {
        std::ofstream o(lbl, std::ios::binary);
        write_be32(o, 0x00000801u);   // magic: unsigned byte, 1 dim
        write_be32(o, count);
        std::vector<char> buf(count);
        for (std::uint32_t n = 0; n < count; ++n) buf[n] = static_cast<char>(n % 10u);
        o.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }
}

struct fixture {
    fs::path dir;
    fs::path img, lbl;
    explicit fixture(std::string_view tag, std::uint32_t count = 8) {
        dir = fs::temp_directory_path() / ("test_mnist_io_manifest_" + std::string(tag));
        std::error_code ec;
        fs::create_directories(dir, ec);
        REQUIRE_FALSE(ec);
        img = dir / "images.idx";
        lbl = dir / "labels.idx";
        write_synthetic_mnist(img, lbl, count);
    }
    fs::path write_manifest(std::string_view name, std::string_view text) const {
        const fs::path p = dir / name;
        std::ofstream o(p);
        REQUIRE(o.good());
        o << text;
        return p;
    }
};

// A clean fixed-order manifest config over the synthetic set.
mnist::mnist_param_t manifest_params(const fixture& fx, const fs::path& manifest) {
    mnist::mnist_param_t p;
    p.image_file = fx.img.string();
    p.label_file = fx.lbl.string();
    p.index_manifest = manifest.string();
    p.change_interval = 1;
    p.noise_range = {0.0, 0.0};
    p.random_seed = seed_t{ 0 };
    p.shuffle = false;
    return p;
}

// Record the source index presented at each of `steps` process() calls,
// driving the loader exactly as the sim thread does.
std::vector<std::size_t> presented_indices(mnist::mnist_io& io, int steps) {
    sim::io_buffer_cpu buf(io.get_outp_size());
    io.set_output_buffer(buf);
    std::vector<std::size_t> seen;
    for (int i = 0; i < steps; ++i) {
        seen.push_back(io.current_index());
        REQUIRE(io.process());
    }
    return seen;
}

} // namespace

TEST_CASE("mnist_io presents samples in index_manifest order") {
    fixture fx{"order"};
    const auto manifest = fx.write_manifest("val.idx", "3 1 4\n");

    mnist::mnist_io io(manifest_params(fx, manifest), sim::output_id_t{0});

    // change_interval=1 means one source index per step; two epochs over the
    // 3-entry manifest must read source indices 3,1,4 then wrap, never the
    // canonical 0,1,2 source order.
    CHECK(presented_indices(io, 6) == std::vector<std::size_t>{3, 1, 4, 3, 1, 4});
    CHECK(io.current_epoch() == 2);
}

TEST_CASE("mnist_io with no index_manifest keeps canonical source order") {
    fixture fx{"legacy"};
    mnist::mnist_param_t p;
    p.image_file = fx.img.string();
    p.label_file = fx.lbl.string();
    // legacy config: no manifest, sequential default order.
    p.change_interval = 1;
    p.noise_range = {0.0, 0.0};
    p.shuffle = false;

    mnist::mnist_io io(p, sim::output_id_t{0});
    CHECK(presented_indices(io, 4) == std::vector<std::size_t>{0, 1, 2, 3});
}

TEST_CASE("mnist_io rejects a missing index_manifest file") {
    fixture fx{"missing"};
    auto p = manifest_params(fx, fx.dir / "does_not_exist.idx");
    CHECK_THROWS_AS((mnist::mnist_io{p, sim::output_id_t{0}}), std::runtime_error);
}

TEST_CASE("mnist_io rejects a manifest index outside the source dataset") {
    fixture fx{"oob"};  // 8 images -> index 8 is out of range
    const auto manifest = fx.write_manifest("bad.idx", "0 8\n");
    auto p = manifest_params(fx, manifest);
    CHECK_THROWS_AS((mnist::mnist_io{p, sim::output_id_t{0}}), std::runtime_error);
}
