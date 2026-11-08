#ifndef COAST_COASTD_SHA256_H
#define COAST_COASTD_SHA256_H

// -----------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), vendored so coastd needs no crypto library: it is used
// only to prove WHICH bytes a file holds (`hash_inputs`), never for anything
// secret, so the plain portable form is enough. Streaming: update() any number of
// times, then digest() once. Tested against the FIPS 180-2 / NIST CAVP vectors in
// tests/unit/coastd/test_sha256.cpp.
// -----------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace coastd {

class sha256 {
public:
    using digest_t = std::array<std::uint8_t, 32>;

    void update(const void* data, std::size_t n) noexcept;
    void update(std::string_view s) noexcept { update(s.data(), s.size()); }
    // Pads and returns the digest. The object is spent afterwards: a further
    // update() or digest() is a caller bug and asserts.
    [[nodiscard]] digest_t digest() noexcept;

private:
    void block(const std::uint8_t* p) noexcept;

    std::array<std::uint32_t, 8> h_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<std::uint8_t, 64> buf_{};
    std::size_t   buf_len_ = 0;
    std::uint64_t total_   = 0;   // bytes; the length field is total_ * 8 bits
    bool          done_    = false;
};

// Lower-case hex, the form `sha256sum` and Python's hexdigest() print.
[[nodiscard]] std::string to_hex(const sha256::digest_t& d);

} // namespace coastd

#endif // COAST_COASTD_SHA256_H
