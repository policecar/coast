#include "daemon/sha256.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>

namespace coastd {

namespace {

constexpr std::array<std::uint32_t, 64> k = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

constexpr std::uint32_t load_be32(const std::uint8_t* p) noexcept {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
           (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}

} // namespace

void sha256::block(const std::uint8_t* p) noexcept {
    std::array<std::uint32_t, 64> w{};
    for (int t = 0; t < 16; ++t) w[t] = load_be32(p + 4 * t);
    for (int t = 16; t < 64; ++t) {
        const std::uint32_t s0 = std::rotr(w[t - 15], 7) ^ std::rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
        const std::uint32_t s1 = std::rotr(w[t - 2], 17) ^ std::rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = h_;
    for (int t = 0; t < 64; ++t) {
        const std::uint32_t t1 = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                                 ((e & f) ^ (~e & g)) + k[t] + w[t];
        const std::uint32_t t2 = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) +
                                 ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void sha256::update(const void* data, std::size_t n) noexcept {
    assert(!done_ && "sha256::update after digest");
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_ += n;
    if (buf_len_ > 0) {   // top up a partial block first
        const std::size_t take = std::min(n, buf_.size() - buf_len_);
        std::memcpy(buf_.data() + buf_len_, p, take);
        buf_len_ += take; p += take; n -= take;
        if (buf_len_ < buf_.size()) return;
        block(buf_.data());
        buf_len_ = 0;
    }
    for (; n >= 64; p += 64, n -= 64) block(p);
    std::memcpy(buf_.data(), p, n);
    buf_len_ = n;
}

sha256::digest_t sha256::digest() noexcept {
    assert(!done_ && "sha256::digest called twice");
    const std::uint64_t bits = total_ * 8;
    const std::uint8_t one = 0x80;
    update(&one, 1);
    const std::uint8_t zero = 0;
    while (buf_len_ != 56) update(&zero, 1);
    std::array<std::uint8_t, 8> len{};
    for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    update(len.data(), len.size());
    assert(buf_len_ == 0);
    done_ = true;
    digest_t out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            out[4 * i + j] = static_cast<std::uint8_t>(h_[i] >> (24 - 8 * j));
    return out;
}

std::string to_hex(const sha256::digest_t& d) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string s(d.size() * 2, '0');
    for (std::size_t i = 0; i < d.size(); ++i) {
        s[2 * i]     = hex[d[i] >> 4];
        s[2 * i + 1] = hex[d[i] & 0xf];
    }
    return s;
}

} // namespace coastd
