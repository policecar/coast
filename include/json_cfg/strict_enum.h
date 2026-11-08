//
// Strict enum <-> JSON mapping for config enums.
//
// Drop-in replacement for NLOHMANN_JSON_SERIALIZE_ENUM with the same table syntax.
// The nlohmann macro silently maps an unrecognised value to the FIRST listed pair,
// so a config typo ("proxymal", "max_sotfmax") or a value from a newer branch runs
// as some other, valid-looking setting. COAST_JSON_STRICT_ENUM throws instead:
//
//   from_json: a JSON value not in the table -> std::invalid_argument naming the
//              enum, the offending value and the accepted values.
//   to_json:   an enumerator not in the table -> std::invalid_argument (a
//              programming error: the table is missing an entry).
//
// This is the one mechanism for all config enums (see the former sharp-edge note
// in ngm25_json.h): every config enum uses it, none hand-rolls its own adaptor.
// Tables carry no `{X::TYPE_COUNT, nullptr}` row: JSON null is rejected like any
// other unknown value, and serialising a sentinel throws. A null-mapped sentinel
// would parse into a value that is not a real setting (dendrite TYPE_COUNT indexes
// past a per-type array; ri_variant TYPE_COUNT leaves every weight 0).
//

#ifndef COAST_STRICT_ENUM_H
#define COAST_STRICT_ENUM_H

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

namespace coast::json_detail {

template <typename Pair, std::size_t N>
std::string accepted_values(const Pair (&m)[N])
{
    std::string out;
    for (const auto& p : m) {
        if (!out.empty()) out += ", ";
        out += p.second.dump();
    }
    return out;
}

}

#define COAST_JSON_STRICT_ENUM(ENUM_TYPE, ...)                                                    \
    template <typename BasicJsonType>                                                             \
    inline void to_json(BasicJsonType& j, const ENUM_TYPE& e)                                     \
    {                                                                                             \
        static_assert(std::is_enum_v<ENUM_TYPE>, #ENUM_TYPE " must be an enum!");                 \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                       \
        const auto it = std::find_if(std::begin(m), std::end(m),                                  \
                                     [e](const auto& p) { return p.first == e; });                \
        if (it == std::end(m))                                                                    \
            throw std::invalid_argument(std::string(#ENUM_TYPE ": no JSON name for value ")       \
                                        + std::to_string(static_cast<long long>(e)));             \
        j = it->second;                                                                           \
    }                                                                                             \
    template <typename BasicJsonType>                                                             \
    inline void from_json(const BasicJsonType& j, ENUM_TYPE& e)                                   \
    {                                                                                             \
        static_assert(std::is_enum_v<ENUM_TYPE>, #ENUM_TYPE " must be an enum!");                 \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                       \
        const auto it = std::find_if(std::begin(m), std::end(m),                                  \
                                     [&j](const auto& p) { return p.second == j; });              \
        if (it == std::end(m))                                                                    \
            throw std::invalid_argument(std::string(#ENUM_TYPE ": unknown value ") + j.dump()     \
                                        + " (expected one of: "                                   \
                                        + ::coast::json_detail::accepted_values(m) + ")");        \
        e = it->first;                                                                            \
    }

// The same mapping, for an enum whose names live in a table elsewhere (e.g.
// sim::entity_types): RANGE is an expression yielding a range of rows, and
// TYPE_MEMBER / NAME_MEMBER name the row's enumerator and its JSON string.
// Same contract as COAST_JSON_STRICT_ENUM.
#define COAST_JSON_STRICT_ENUM_RANGE(ENUM_TYPE, RANGE, TYPE_MEMBER, NAME_MEMBER)                  \
    template <typename BasicJsonType>                                                             \
    inline void to_json(BasicJsonType& j, const ENUM_TYPE& e)                                     \
    {                                                                                             \
        static_assert(std::is_enum_v<ENUM_TYPE>, #ENUM_TYPE " must be an enum!");                 \
        for (const auto& row : RANGE)                                                             \
            if (row.TYPE_MEMBER == e) { j = row.NAME_MEMBER; return; }                            \
        throw std::invalid_argument(std::string(#ENUM_TYPE ": no JSON name for value ")           \
                                    + std::to_string(static_cast<long long>(e)));                 \
    }                                                                                             \
    template <typename BasicJsonType>                                                             \
    inline void from_json(const BasicJsonType& j, ENUM_TYPE& e)                                   \
    {                                                                                             \
        static_assert(std::is_enum_v<ENUM_TYPE>, #ENUM_TYPE " must be an enum!");                 \
        std::string accepted;                                                                     \
        for (const auto& row : RANGE) {                                                           \
            const BasicJsonType name = row.NAME_MEMBER;                                           \
            if (name == j) { e = row.TYPE_MEMBER; return; }                                       \
            if (!accepted.empty()) accepted += ", ";                                              \
            accepted += name.dump();                                                              \
        }                                                                                         \
        throw std::invalid_argument(std::string(#ENUM_TYPE ": unknown value ") + j.dump()         \
                                    + " (expected one of: " + accepted + ")");                    \
    }

#endif // COAST_STRICT_ENUM_H
