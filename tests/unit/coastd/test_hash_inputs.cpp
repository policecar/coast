#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/hash_inputs.h"

using coastd::hash_inputs;
using coastd::hash_limits;
namespace fs = std::filesystem;

// hash_inputs is what lets a remote client prove which bytes a run's ./data/<rel>
// read. Its one job besides hashing is to refuse any path that is not a file under
// dataset_base — every refusal below is a way a client could otherwise read (or
// probe for) a file elsewhere on the daemon's host.

namespace {
constexpr const char* abc_sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

struct scratch {
    fs::path root, base;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() / ("coastd_hi_" + std::to_string(::getpid()) + "_" +
                                            std::to_string(ctr.fetch_add(1)));
        base = root / "data";
        fs::create_directories(base / "mnist");
        write(base / "mnist" / "a", "abc");
        write(base / "b", "");
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
    static void write(const fs::path& p, std::string_view s) { std::ofstream(p, std::ios::binary) << s; }
};

// The refusal names the path and says why; never a partial answer.
void refused(const fs::path& base, std::vector<std::string> rels, std::string_view why,
             const hash_limits& lim = {}) {
    const auto r = hash_inputs(base, rels, lim);
    REQUIRE_FALSE(r.has_value());
    CAPTURE(r.error());
    CHECK(r.error().find(why) != std::string::npos);
}
} // namespace

TEST_CASE("hash_inputs: hashes regular files relative to dataset_base") {
    scratch s;
    const auto r = hash_inputs(s.base, {"mnist/a", "b", "./mnist/a"});
    REQUIRE_MESSAGE(r.has_value(), r.error());
    CHECK(r->size() == 3);
    CHECK(r->at("mnist/a") == abc_sha);
    CHECK(r->at("./mnist/a") == abc_sha);   // keyed by the path as given
    CHECK(r->at("b") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("hash_inputs: a symlink that stays inside dataset_base is followed") {
    scratch s;
    fs::create_symlink("mnist/a", s.base / "link");
    const auto r = hash_inputs(s.base, {"link"});
    REQUIRE_MESSAGE(r.has_value(), r.error());
    CHECK(r->at("link") == abc_sha);
}

TEST_CASE("hash_inputs: dataset_base may itself be a symlink") {
    scratch s;
    fs::create_directory_symlink(s.base, s.root / "via");
    const auto r = hash_inputs(s.root / "via", {"mnist/a"});
    REQUIRE_MESSAGE(r.has_value(), r.error());
    CHECK(r->at("mnist/a") == abc_sha);
}

TEST_CASE("hash_inputs: refuses paths that leave dataset_base") {
    scratch s;
    scratch::write(s.root / "secret", "x");
    fs::create_directories(s.root / "data2");
    scratch::write(s.root / "data2" / "f", "x");

    SUBCASE("absolute") { refused(s.base, {(s.base / "b").string()}, "is absolute"); }
    SUBCASE("'..' even when it would land back inside") {
        refused(s.base, {"mnist/../b"}, "contains '..'");
        refused(s.base, {"../secret"}, "contains '..'");
    }
    SUBCASE("a symlink escaping dataset_base") {
        fs::create_symlink(s.root / "secret", s.base / "esc");
        refused(s.base, {"esc"}, "outside dataset_base");
    }
    SUBCASE("a sibling whose name has dataset_base as a string prefix") {
        // /…/data2/f starts with the string /…/data — a string-prefix test would pass it.
        fs::create_symlink(s.root / "data2" / "f", s.base / "sib");
        refused(s.base, {"sib"}, "outside dataset_base");
    }
}

TEST_CASE("hash_inputs: refuses what is not a readable regular file") {
    scratch s;
    SUBCASE("missing")        { refused(s.base, {"mnist/nope"}, "mnist/nope"); }
    SUBCASE("a directory")    { refused(s.base, {"mnist"}, "not a regular file"); }
    SUBCASE("the base itself"){ refused(s.base, {"."}, "not a regular file"); }
    SUBCASE("empty string")   { refused(s.base, {""}, "empty path"); }
    SUBCASE("one bad path refuses the whole request") {
        refused(s.base, {"mnist/a", "mnist/nope"}, "mnist/nope");
    }
}

TEST_CASE("hash_inputs: refuses a missing or unset dataset_base and an empty request") {
    scratch s;
    refused({}, {"b"}, "not configured");
    refused(s.root / "not_mounted", {"b"}, "dataset_base");
    refused(s.base, {}, "no paths");
}

TEST_CASE("hash_inputs: the caps bound one request") {
    scratch s;   // mnist/a is 3 bytes, b is 0
    SUBCASE("path count") {
        CHECK(hash_inputs(s.base, {"mnist/a", "b"}, {.max_paths = 2}).has_value());
        refused(s.base, {"mnist/a", "b"}, "exceed the cap", {.max_paths = 1});
    }
    SUBCASE("total bytes, checked before any file is read") {
        CHECK(hash_inputs(s.base, {"mnist/a", "b"}, {.max_total_bytes = 3}).has_value());
        refused(s.base, {"mnist/a", "mnist/a"}, "exceed the cap", {.max_total_bytes = 5});
    }
}
