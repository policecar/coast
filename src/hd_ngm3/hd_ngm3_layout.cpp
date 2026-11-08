//
// Layout check for one ngm3 bank (see hd_ngm3_layout.h for the laws).
//

#include "hd_ngm3/hd_ngm3_layout.h"

#include <bit>
#include <cmath>
#include <format>

namespace ngm3 {

bool check_bank_layout(std::span<const float>        permanences,
                       std::span<const std::uint8_t> segment_ids,
                       std::span<const float>        adapt_history,
                       std::span<const float>        mismatch,
                       std::size_t                   neuron_cnt,
                       std::size_t                   d_cnt,
                       std::uint8_t                  branch_depth,
                       std::size_t                   input_size,
                       std::string                  *why)
{
    auto fail = [&](std::string msg) {
        if (why) *why = std::move(msg);
        return false;
    };

    const unsigned    depth = branch_depth;
    const std::size_t rows  = std::size_t{1} << depth;
    const std::size_t total = neuron_cnt * d_cnt * rows * input_size;
    if (permanences.size() != total || segment_ids.size() != total ||
        adapt_history.size() != total || mismatch.size() != total)
        return fail(std::format("array sizes {}/{}/{}/{} (permanences/segment_ids/"
                                "adapt_history/mismatch), expected {}",
                                permanences.size(), segment_ids.size(),
                                adapt_history.size(), mismatch.size(), total));

    // L1: the values the compute can hold. The bitwise replica compare (L5)
    // cannot see a NaN that is copied into every row of its run, and the
    // response and adapt sweeps would carry it into every output.
    for (std::size_t i = 0; i < total; ++i) {
        const char *bad = nullptr;
        if (!(permanences[i] >= 0.0f && permanences[i] <= 1.0f))
            bad = "permanence outside [0, 1]";
        else if (!(std::isfinite(adapt_history[i]) && adapt_history[i] >= 0.0f))
            bad = "adapt_history not finite and >= 0";
        else if (!std::isfinite(mismatch[i]))
            bad = "mismatch not finite";
        if (bad != nullptr) {
            const std::size_t per_nd = rows * input_size;
            const std::size_t nd = i / per_nd, rem = i % per_nd;
            return fail(std::format("neuron {} dendrite {} input {} row {}: {} "
                                    "(permanence {}, adapt_history {}, mismatch {}) (L1)",
                                    nd / d_cnt, nd % d_cnt, rem % input_size, rem / input_size,
                                    bad, permanences[i], adapt_history[i], mismatch[i]));
        }
    }

    const unsigned max_seg = (2u << depth) - 1u;
    auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };

    for (std::size_t nd = 0; nd < neuron_cnt * d_cnt; ++nd) {
        const std::size_t block = nd * rows * input_size;
        for (std::size_t k = 0; k < input_size; ++k) {
            auto at = [&](std::size_t r) { return block + r * input_size + k; };
            auto where = [&](std::size_t r) {
                return std::format("neuron {} dendrite {} input {} row {}",
                                   nd / d_cnt, nd % d_cnt, k, r);
            };
            // Walk the column run by run: each run must start where its
            // segment's interval starts, so the runs tile [0, R) exactly.
            for (std::size_t r = 0; r < rows; ) {
                const unsigned s = segment_ids[at(r)];
                if (s < 1 || s > max_seg)
                    return fail(std::format("{}: segment id {} outside [1, {}] (L2)",
                                            where(r), s, max_seg));
                const unsigned    shift = depth - (static_cast<unsigned>(std::bit_width(s)) - 1u);
                const std::size_t begin = (std::size_t{s} << shift) - rows;
                const std::size_t len   = std::size_t{1} << shift;
                if (begin != r)
                    return fail(std::format("{}: segment {} belongs to rows [{}, {}) (L3)",
                                            where(r), s, begin, begin + len));
                const std::size_t head = at(r);
                for (std::size_t q = r + 1; q < r + len; ++q) {
                    const std::size_t i = at(q);
                    if (segment_ids[i] != s)
                        return fail(std::format("{}: segment {} does not fill rows [{}, {}), "
                                                "row {} holds {} (L4)",
                                                where(r), s, r, r + len, q,
                                                static_cast<unsigned>(segment_ids[i])));
                    if (bits(permanences[i])   != bits(permanences[head]) ||
                        bits(mismatch[i])      != bits(mismatch[head])    ||
                        bits(adapt_history[i]) != bits(adapt_history[head]))
                        return fail(std::format("{}: replica row {} of segment {} differs "
                                                "from row {} (L5)", where(r), q, s, r));
                }
                r += len;
            }
        }
    }
    return true;
}

} // namespace ngm3
