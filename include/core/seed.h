//
// Random seeds and seed offsets (issue #167).
//
// Every RNG in COAST takes a 32-bit seed: mt19937_simd and mt11213b_simd,
// ngm2::stream_seed's result, and the u32 seeds and keys of the WGSL shaders. A
// seed is therefore a uint32, end to end, so the value an entity reports via
// effective_seed() is the value its RNG was seeded with.
//
// seed_t and seed_offset_t are distinct strong types. Nothing converts to them
// implicitly, so the compiler finds every place a seed is produced or consumed,
// and a seed and an offset cannot be swapped. The raw value is
// std::to_underlying(s).
//
// A seed with offsets applied is composed modulo 2^32 (unsigned arithmetic, so
// well-defined). Distinct (seed, offset) pairs can land on the same seed; that is
// accepted — the composition is deterministic, which is what reproducibility needs.
//
// Input (config JSON, --seed-offset): an integer in [0, 2^32 - 1], the value the
// RNG is seeded with. Anything else (a negative or fractional number, a string) is
// refused; a config names it by its JSON path (json_loader checks every seed key up
// front). Negative input used to be taken as its two's complement (#167); that
// made the stored seed differ from the written one, which json_loader's
// round-trip check now reports as a misread, so it is refused instead (#199).
//

#ifndef COAST_SEED_H
#define COAST_SEED_H

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace core {

enum class seed_t        : std::uint32_t {};
enum class seed_offset_t : std::uint32_t {};

[[nodiscard]] constexpr seed_t operator+(seed_t s, seed_offset_t o) noexcept
{
    return seed_t{ static_cast<std::uint32_t>(std::to_underlying(s) + std::to_underlying(o)) };
}

[[nodiscard]] constexpr seed_offset_t operator+(seed_offset_t a, seed_offset_t b) noexcept
{
    return seed_offset_t{ static_cast<std::uint32_t>(std::to_underlying(a) + std::to_underlying(b)) };
}

// The input integer as a seed, or nullopt outside [0, 2^32 - 1].
[[nodiscard]] constexpr std::optional<std::uint32_t> seed_bits(std::int64_t v) noexcept
{
    if (v < 0 || v > std::numeric_limits<std::uint32_t>::max())
        return std::nullopt;
    return static_cast<std::uint32_t>(v);
}

// A JSON value as a seed, or nullopt unless it is an integer in [0, 2^32 - 1]. Shared by the hooks below and by json_loader, which checks
// every seed key of a config with it before binding, so a refused seed is named by
// its JSON path (the hooks cannot see their key).
template <typename BasicJsonType>
[[nodiscard]] std::optional<std::uint32_t> seed_bits_of(const BasicJsonType& j)
{
    if (j.is_number_unsigned()) {   // checked first: nlohmann's is_number_integer() includes it
        const auto v = j.template get<std::uint64_t>();
        if (v > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        return static_cast<std::uint32_t>(v);
    }
    if (j.is_number_integer()) return seed_bits(j.template get<std::int64_t>());
    return std::nullopt;            // float, string, bool, null, ...
}

inline constexpr const char* k_seed_rule = "must be an integer in [0, 4294967295]";

namespace detail {

template <typename BasicJsonType>
std::uint32_t seed_bits_or_throw(const BasicJsonType& j, const char* what)
{
    if (const auto bits = seed_bits_of(j)) return *bits;
    throw std::invalid_argument(std::string(what) + " " + k_seed_rule + ", got " + j.dump());
}

} // namespace detail

// nlohmann/json hooks, found by ADL. Templates over BasicJsonType so they serve
// both nlohmann::json and ordered_json; more specialised than nlohmann's generic
// enum overloads, so they win wherever this header is visible — which is wherever
// seed_t is.
template <typename BasicJsonType>
void from_json(const BasicJsonType& j, seed_t& s)
{ s = seed_t{ detail::seed_bits_or_throw(j, "a random seed") }; }

template <typename BasicJsonType>
void from_json(const BasicJsonType& j, seed_offset_t& o)
{ o = seed_offset_t{ detail::seed_bits_or_throw(j, "a random seed offset") }; }

template <typename BasicJsonType>
void to_json(BasicJsonType& j, seed_t s) { j = std::to_underlying(s); }

template <typename BasicJsonType>
void to_json(BasicJsonType& j, seed_offset_t o) { j = std::to_underlying(o); }

} // namespace core

using core::seed_t;
using core::seed_offset_t;

#endif // COAST_SEED_H
