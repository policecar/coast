#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>     // getpid for scratch-dir uniqueness

#include <nlohmann/json.hpp>

#include "blob_writer.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// A scratch directory unique to each TEST_CASE invocation. Wiped on destruction.
struct scratch_dir {
    fs::path path;
    explicit scratch_dir(std::string_view label) {
        path = fs::temp_directory_path() /
               ("coast_blob_writer_" + std::string(label) + "_" +
                std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~scratch_dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

json read_blob(const fs::path& p) {
    std::ifstream in(p);
    REQUIRE(in.good());
    std::stringstream buf;
    buf << in.rdbuf();
    return json::parse(buf.str());
}

} // namespace

TEST_CASE("blob_writer: scalar round-trip") {
    scratch_dir sd{"scalar"};
    const auto blob_path = sd.path / "blob.json";

    {
        coast::test::blob_writer w(blob_path);
        w.scalar("temperature", 3.14);
        w.scalar("count", std::int64_t{42});
        w.close();
    }

    auto j = read_blob(blob_path);
    CHECK(j["temperature"].get<double>() == doctest::Approx(3.14));
    CHECK(j["count"].get<std::int64_t>() == 42);
}

TEST_CASE("blob_writer: inline array round-trip") {
    scratch_dir sd{"array"};
    const auto blob_path = sd.path / "blob.json";

    const std::array<double, 4>       d{1.0, 2.0, 3.0, 4.5};
    const std::array<std::int64_t, 3> i{-1, 0, 1};
    {
        coast::test::blob_writer w(blob_path);
        w.array("doubles", std::span<const double>{d});
        w.array("ints",    std::span<const std::int64_t>{i});
    } // dtor closes

    auto j = read_blob(blob_path);
    CHECK(j["doubles"].size() == 4);
    CHECK(j["doubles"][3].get<double>() == doctest::Approx(4.5));
    CHECK(j["ints"][0].get<std::int64_t>() == -1);
    CHECK(j["ints"][2].get<std::int64_t>() == 1);
}

TEST_CASE("blob_writer: nested paths via '/' build sub-objects") {
    scratch_dir sd{"nested"};
    const auto blob_path = sd.path / "blob.json";

    {
        coast::test::blob_writer w(blob_path);
        w.scalar("group_0/dend_3/activity", 0.5);
        w.scalar("group_0/dend_3/spikes", std::int64_t{17});
        w.scalar("group_1/dend_0/activity", 0.25);
        w.close();
    }

    auto j = read_blob(blob_path);
    CHECK(j["group_0"]["dend_3"]["activity"].get<double>() == doctest::Approx(0.5));
    CHECK(j["group_0"]["dend_3"]["spikes"].get<std::int64_t>() == 17);
    CHECK(j["group_1"]["dend_0"]["activity"].get<double>() == doctest::Approx(0.25));
}

TEST_CASE("blob_writer: binary sidecar (doubles)") {
    scratch_dir sd{"bin_f64"};
    const auto blob_path = sd.path / "blob.json";

    const std::array<double, 5> v{1.0, 2.0, 3.0, 4.0, 5.0};
    {
        coast::test::blob_writer w(blob_path);
        w.array_binary("weights", std::span<const double>{v}, "weights.bin");
    }

    auto j = read_blob(blob_path);
    CHECK(j["weights"]["binary"].get<std::string>() == "weights.bin");
    CHECK(j["weights"]["dtype"].get<std::string>()  == "f64");
    CHECK(j["weights"]["shape"][0].get<std::size_t>() == 5u);

    // Sidecar contents.
    std::ifstream in(sd.path / "weights.bin", std::ios::binary);
    REQUIRE(in.good());
    std::array<double, 5> got{};
    in.read(reinterpret_cast<char*>(got.data()),
            static_cast<std::streamsize>(got.size() * sizeof(double)));
    REQUIRE(in.gcount() == static_cast<std::streamsize>(got.size() * sizeof(double)));
    for (std::size_t k = 0; k < got.size(); ++k) {
        CHECK(got[k] == doctest::Approx(v[k]));
    }
}

TEST_CASE("blob_writer: binary sidecar (int64)") {
    scratch_dir sd{"bin_i64"};
    const auto blob_path = sd.path / "blob.json";

    const std::array<std::int64_t, 3> v{-7, 0, 7};
    {
        coast::test::blob_writer w(blob_path);
        w.array_binary("counts", std::span<const std::int64_t>{v}, "c.bin");
    }

    auto j = read_blob(blob_path);
    CHECK(j["counts"]["dtype"].get<std::string>() == "i64");
    CHECK(j["counts"]["shape"][0].get<std::size_t>() == 3u);

    std::ifstream in(sd.path / "c.bin", std::ios::binary);
    REQUIRE(in.good());
    std::array<std::int64_t, 3> got{};
    in.read(reinterpret_cast<char*>(got.data()),
            sizeof(got));
    REQUIRE(in.gcount() == sizeof(got));
    CHECK(got[0] == -7);
    CHECK(got[2] ==  7);
}

TEST_CASE("blob_writer: duplicate field throws") {
    scratch_dir sd{"dup"};
    coast::test::blob_writer w(sd.path / "blob.json");
    w.scalar("x", 1.0);
    CHECK_THROWS_AS(w.scalar("x", 2.0), std::runtime_error);
    CHECK_THROWS_AS(w.scalar("x", std::int64_t{2}), std::runtime_error);
}

TEST_CASE("blob_writer: empty name and empty segment throw") {
    scratch_dir sd{"empty_seg"};
    coast::test::blob_writer w(sd.path / "blob.json");
    CHECK_THROWS_AS(w.scalar("",     1.0), std::runtime_error);
    CHECK_THROWS_AS(w.scalar("a//b", 1.0), std::runtime_error);
    CHECK_THROWS_AS(w.scalar("/x",   1.0), std::runtime_error);
    CHECK_THROWS_AS(w.scalar("x/",   1.0), std::runtime_error);
}

TEST_CASE("blob_writer: write after close() throws") {
    scratch_dir sd{"after_close"};
    coast::test::blob_writer w(sd.path / "blob.json");
    w.scalar("a", 1.0);
    w.close();
    CHECK_THROWS_AS(w.scalar("b", 2.0), std::runtime_error);
    // close() is idempotent.
    CHECK_NOTHROW(w.close());
}

TEST_CASE("blob_writer: nested path collides with scalar throws") {
    scratch_dir sd{"collide"};
    coast::test::blob_writer w(sd.path / "blob.json");
    w.scalar("g/a", 1.0);
    CHECK_THROWS_AS(w.scalar("g/a/inner", 2.0), std::runtime_error);
}

TEST_CASE("blob_writer: sidecar name must be flat") {
    scratch_dir sd{"sidecar_path"};
    coast::test::blob_writer w(sd.path / "blob.json");
    const std::array<double, 1> v{1.0};
    CHECK_THROWS_AS(w.array_binary("a", std::span<const double>{v}, ""),
                    std::runtime_error);
    CHECK_THROWS_AS(w.array_binary("b", std::span<const double>{v}, "sub/x.bin"),
                    std::runtime_error);
}

TEST_CASE("blob_writer: creates parent directories on close()") {
    scratch_dir sd{"mkdir"};
    const auto nested = sd.path / "a" / "b" / "c" / "blob.json";
    {
        coast::test::blob_writer w(nested);
        w.scalar("x", 1.0);
    }
    CHECK(fs::exists(nested));
}
