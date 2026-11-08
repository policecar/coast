#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// #200: dataset loaders must reject input files that do not match what their
// header (IDX) or format (CIFAR) declares, instead of silently presenting zeros
// or writing out of bounds. Each case writes a small synthetic file, corrupts
// it in one way and checks the loader throws; a matching well-formed file is
// checked to load, so a throw cannot come from a broken fixture.

#include "io/mnist_db.h"
#include "io/color_mnist_io.h"
#include "io/cifar10_io.h"
#include "io/cifar100_io.h"
#include "sim_core/io_buffer.h"
#include "logging/mem_state_io.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path scratch_dir(std::string_view tag) {
    const fs::path d = fs::temp_directory_path() / ("test_loader_input_validation_" + std::string(tag));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

void write_be32(std::ostream& os, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
        static_cast<unsigned char>(v >> 8),  static_cast<unsigned char>(v)};
    os.write(reinterpret_cast<const char*>(b), 4);
}

// IDX file: data type `type` (default u8), header dims `dims`, followed by
// `payload` bytes of data. `header_dims` < dims.size() writes a header that
// claims dims.size() dimensions but stops early (truncated header).
void write_idx(const fs::path& p, const std::vector<std::uint32_t>& dims,
               std::size_t payload, std::size_t header_dims = SIZE_MAX,
               std::uint32_t type = 0x08) {
    std::ofstream o(p, std::ios::binary);
    write_be32(o, (type << 8) | static_cast<std::uint32_t>(dims.size()));
    for (std::size_t i = 0; i < std::min(header_dims, dims.size()); ++i) write_be32(o, dims[i]);
    if (header_dims < dims.size()) return;
    const std::vector<char> data(payload, 7);
    o.write(data.data(), static_cast<std::streamsize>(data.size()));
}

// --- IDX ---------------------------------------------------------------------

TEST_CASE("idx_file accepts a file whose data matches its header") {
    const auto d = scratch_dir("idx_ok");
    write_idx(d / "a.idx", {3, 4, 5}, 3 * 4 * 5);
    const mnist::idx_file f((d / "a.idx").string());
    CHECK(f.get_num_elements() == 3);
    CHECK(f.get_element_size() == 20);
    CHECK(f.get_u8().size() == 60);
}

TEST_CASE("idx_file rejects a truncated header") {
    const auto d = scratch_dir("idx_short_header");
    write_idx(d / "a.idx", {3, 4, 5}, 0, /*header_dims=*/2);
    CHECK_THROWS_AS(mnist::idx_file((d / "a.idx").string()), std::runtime_error);
}

TEST_CASE("idx_file rejects data shorter than the header declares") {
    const auto d = scratch_dir("idx_short_data");
    write_idx(d / "a.idx", {3, 4, 5}, 3 * 4 * 5 - 1);
    CHECK_THROWS_AS(mnist::idx_file((d / "a.idx").string()), std::runtime_error);
}

TEST_CASE("idx_file rejects data longer than the header declares") {
    const auto d = scratch_dir("idx_long_data");
    write_idx(d / "a.idx", {3, 4, 5}, 3 * 4 * 5 + 1);
    CHECK_THROWS_AS(mnist::idx_file((d / "a.idx").string()), std::runtime_error);
}

TEST_CASE("idx_file rejects a label file shorter than its header declares") {
    // Label files go through the same reader; before #200 only the image file's
    // data length was checked.
    const auto d = scratch_dir("idx_short_labels");
    write_idx(d / "img.idx", {4, 28, 28}, 4 * 28 * 28);
    write_idx(d / "lbl.idx", {4}, 3);
    CHECK_THROWS_AS(mnist::mnist_db((d / "img.idx").string(), (d / "lbl.idx").string()),
                    std::runtime_error);
}

TEST_CASE("idx_file rejects sizes whose product overflows") {
    // Each product wraps to exactly 0 mod 2^64, so an empty payload would pass
    // the data-size check: only the overflow guards reject these files.
    const auto d = scratch_dir("idx_overflow");
    // element_size = 2^16^4 = 2^64
    write_idx(d / "elem.idx", {1, 65536, 65536, 65536, 65536}, 0);
    CHECK_THROWS_AS(mnist::idx_file((d / "elem.idx").string()), std::runtime_error);
    // num_elements * element_size = 2^16 * 2^48 = 2^64
    write_idx(d / "total.idx", {65536, 65536, 65536, 65536}, 0);
    CHECK_THROWS_AS(mnist::idx_file((d / "total.idx").string()), std::runtime_error);
}

TEST_CASE("idx_file rejects a header declaring more data than the file can hold") {
    // 2^31 * 2^31 = 2^62 bytes: must fail on the size check, not in the allocator.
    const auto d = scratch_dir("idx_huge");
    write_idx(d / "a.idx", {1u << 31, 1u << 31}, 16);
    CHECK_THROWS_AS(mnist::idx_file((d / "a.idx").string()), std::runtime_error);
}

TEST_CASE("idx_file rejects IDX data types other than unsigned byte") {
    // A well-formed float32 (0x0D) file: 3 values = 12 bytes. Only 0x08 is consumed
    // by any loader, so the other types are rejected rather than mis-read.
    const auto d = scratch_dir("idx_type");
    write_idx(d / "a.idx", {3}, 3 * 4, SIZE_MAX, 0x0D);
    CHECK_THROWS_AS(mnist::idx_file((d / "a.idx").string()), std::runtime_error);
}

TEST_CASE("mnist_db rejects a dataset with no image data") {
    // Well-formed IDX, but nothing to sample: before this check the first
    // process() threw out_of_range from get_norm_image and aborted the run.
    const auto d = scratch_dir("idx_empty");
    write_idx(d / "img0.idx", {0, 28, 28}, 0);
    write_idx(d / "lbl0.idx", {0}, 0);
    CHECK_THROWS_AS(mnist::mnist_db((d / "img0.idx").string(), (d / "lbl0.idx").string()),
                    std::runtime_error);
    // Images of size 0: two images, but each one empty.
    write_idx(d / "imgz.idx", {2, 0, 28}, 0);
    write_idx(d / "lblz.idx", {2}, 2);
    CHECK_THROWS_AS(mnist::mnist_db((d / "imgz.idx").string(), (d / "lblz.idx").string()),
                    std::runtime_error);
    // Control: the same shape with one image loads.
    write_idx(d / "img1.idx", {1, 28, 28}, 28 * 28);
    write_idx(d / "lbl1.idx", {1}, 1);
    CHECK_NOTHROW(mnist::mnist_db((d / "img1.idx").string(), (d / "lbl1.idx").string()));
}

// --- color_mnist -------------------------------------------------------------

color_mnist::color_mnist_param_t color_params(const fs::path& img, const fs::path& lbl) {
    color_mnist::color_mnist_param_t p;
    p.image_file = img.string();
    p.label_file = lbl.string();
    p.change_interval = 1;
    p.noise_range = {0.0, 0.0};
    return p;
}

TEST_CASE("color_mnist accepts 28x28 images and rejects other sizes") {
    const auto d = scratch_dir("color_mnist");
    write_idx(d / "lbl.idx", {2}, 2);
    write_idx(d / "img28.idx", {2, 28, 28}, 2 * 28 * 28);
    write_idx(d / "img32.idx", {2, 32, 32}, 2 * 32 * 32);
    CHECK_NOTHROW(color_mnist::color_mnist_io(color_params(d / "img28.idx", d / "lbl.idx"),
                                              sim::output_id_t{0}));
    // 32x32 would make process() write 3*1024 floats into a 3*784 output span.
    CHECK_THROWS_AS(color_mnist::color_mnist_io(color_params(d / "img32.idx", d / "lbl.idx"),
                                                sim::output_id_t{0}),
                    std::runtime_error);
}

TEST_CASE("color_mnist refuses a checkpoint colour outside red/green/blue") {
    // #238 sibling: current_color indexes the RGB channel (outp[i*3 + c]) and
    // color_names, and is only redrawn at the next interval start, so a restored
    // value outside [0, 2] wrote past the pixel's three channels.
    const auto d = scratch_dir("color_mnist_state");
    write_idx(d / "lbl.idx", {2}, 2);
    write_idx(d / "img.idx", {2, 28, 28}, 2 * 28 * 28);
    color_mnist::color_mnist_io io(color_params(d / "img.idx", d / "lbl.idx"),
                                   sim::output_id_t{0});
    logging::mem_state_writer w;
    io.save_state(w);
    auto tree = w.take_root();
    for (const std::int32_t c : {-1, 3, 7}) {
        CAPTURE(c);
        tree.fields["current_color"] = c;
        logging::mem_state_reader r(tree, logging::k_format_version);
        r.set_continues_run(true);   // a resume: the colour is restored
        CHECK_THROWS_AS(io.load_state(r), std::runtime_error);
    }
    // Control: every real colour loads.
    for (const std::int32_t c : {0, 1, 2}) {
        CAPTURE(c);
        tree.fields["current_color"] = c;
        logging::mem_state_reader r(tree, logging::k_format_version);
        r.set_continues_run(true);   // a resume: the colour is restored
        CHECK_NOTHROW(io.load_state(r));
    }
}

// --- CIFAR -------------------------------------------------------------------

constexpr std::size_t cifar_pixels = 32 * 32;

// `samples` CIFAR records with `label_bytes` label bytes each, plus `extra`
// trailing bytes (negative: cut that many bytes off the end). Pixel byte for
// sample n, channel c, pixel p is (n + 3*c + p) & 0xFF.
void write_cifar(const fs::path& p, std::size_t label_bytes, std::size_t samples,
                 std::ptrdiff_t extra = 0) {
    std::vector<char> data;
    for (std::size_t n = 0; n < samples; ++n) {
        for (std::size_t l = 0; l < label_bytes; ++l) data.push_back(static_cast<char>(n + l));
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t px = 0; px < cifar_pixels; ++px)
                data.push_back(static_cast<char>((n + 3 * c + px) & 0xFFu));
    }
    if (extra > 0) data.insert(data.end(), static_cast<std::size_t>(extra), 0);
    else data.erase(data.end() + extra, data.end());
    std::ofstream o(p, std::ios::binary);
    o.write(data.data(), static_cast<std::streamsize>(data.size()));
}

cifar100::cifar100_param_t cifar100_params(const fs::path& dir) {
    cifar100::cifar100_param_t p;
    p.data_dir = dir.string();
    p.use_test_set = true;
    p.change_interval = 1;
    p.noise_range = {0.0, 0.0};
    p.shuffle = false;
    return p;
}

TEST_CASE("cifar100 loads a complete file and converts planar to interleaved RGB") {
    const auto d = scratch_dir("cifar100_ok");
    write_cifar(d / "test.bin", 2, 10000);
    const cifar100::cifar100_io io(cifar100_params(d), sim::output_id_t{0});
    REQUIRE(io.sample_count() == 10000);
    for (const std::size_t n : {std::size_t{0}, std::size_t{4321}, std::size_t{9999}}) {
        const auto s = io.get_sample(n);
        REQUIRE(s.size() == cifar_pixels * 3);
        for (const std::size_t px : {std::size_t{0}, std::size_t{17}, cifar_pixels - 1})
            for (std::size_t c = 0; c < 3; ++c)
                CHECK(s[px * 3 + c] ==
                      static_cast<float>((n + 3 * c + px) & 0xFFu) / 255.0f);
    }
}

TEST_CASE("cifar100 rejects a truncated file") {
    // Before #200 the missing tail became all-black samples, silently.
    const auto d = scratch_dir("cifar100_short");
    write_cifar(d / "test.bin", 2, 10000, -1);
    CHECK_THROWS_AS(cifar100::cifar100_io(cifar100_params(d), sim::output_id_t{0}),
                    std::runtime_error);
}

TEST_CASE("cifar100 rejects a file with trailing data") {
    const auto d = scratch_dir("cifar100_long");
    write_cifar(d / "test.bin", 2, 10000, 1);
    CHECK_THROWS_AS(cifar100::cifar100_io(cifar100_params(d), sim::output_id_t{0}),
                    std::runtime_error);
}

TEST_CASE("cifar10 rejects a truncated test batch and loads a complete one") {
    const auto d = scratch_dir("cifar10");
    cifar10::cifar10_param_t p;
    p.data_dir = d.string();
    p.use_test_set = true;
    p.change_interval = 1;
    p.noise_range = {0.0, 0.0};
    p.shuffle = false;

    write_cifar(d / "test_batch.bin", 1, 10000, -100);
    CHECK_THROWS_AS(cifar10::cifar10_io(p, sim::output_id_t{0}), std::runtime_error);

    write_cifar(d / "test_batch.bin", 1, 10000);
    const cifar10::cifar10_io io(p, sim::output_id_t{0});
    CHECK(io.sample_count() == 10000);
    CHECK(io.get_sample(9999)[2] == static_cast<float>((9999 + 6) & 0xFFu) / 255.0f);
}

}  // namespace
