#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstddef>
#include <string>
#include <vector>

#include "daemon/sha256.h"

using coastd::sha256;

namespace {
std::string hex_of(std::string_view s) {
    sha256 h;
    h.update(s);
    return coastd::to_hex(h.digest());
}
std::string pattern(std::size_t n) {   // bytes 0,1,…,250,0,1,… — not a repeated char
    std::string s(n, '\0');
    for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<char>(i % 251);
    return s;
}
} // namespace

TEST_CASE("sha256: FIPS 180-2 vectors") {
    CHECK(hex_of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex_of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(hex_of("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                 "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu") ==
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    CHECK(hex_of(std::string(1'000'000, 'a')) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("sha256: lengths around the padding boundaries (55/56 and 64 bytes per block)") {
    // Expected values: Python hashlib.sha256 over the same pattern.
    const std::vector<std::pair<std::size_t, const char*>> cases = {
        {  1, "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
        { 55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        { 56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        { 57, "2fe741af801cc238602ac0ec6a7b0c3a8a87c7fc7d7f02a3fe03d1c12eac4d8f"},
        { 63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        { 64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        { 65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        {111, "60780e9451bdc43cf4530ffc95cbb0c4eb24dae2c39f55f334d679e076c08065"},
        {112, "09373f127d34e61dbbaa8bc4499c87074f2ddb10e1b465f506d7d70a15011979"},
        {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"},
        {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
        {127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
        {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
        {129, "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135"},
    };
    for (const auto& [n, want] : cases) {
        CAPTURE(n);
        CHECK(hex_of(pattern(n)) == want);
    }
}

TEST_CASE("sha256: the digest does not depend on how the input is split into updates") {
    const std::string msg = pattern(1000);
    const std::string whole = hex_of(msg);
    for (const std::size_t chunk : {1u, 3u, 55u, 63u, 64u, 65u, 127u, 999u}) {
        CAPTURE(chunk);
        sha256 h;
        for (std::size_t i = 0; i < msg.size(); i += chunk)
            h.update(msg.data() + i, std::min(chunk, msg.size() - i));
        CHECK(coastd::to_hex(h.digest()) == whole);
    }
}
