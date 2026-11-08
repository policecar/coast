//
// Layout and value check for one ngm3 bank's dense arrays — shared by the CPU
// and GPU groups' load_state, so a checkpoint that breaks the replica layout or
// holds a value the compute cannot (NaN, a permanence outside [0, 1]) is
// refused before it can be stepped.
//

#ifndef HD_NGM3_LAYOUT_H
#define HD_NGM3_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ngm3 {

// A bank is [neuron_cnt][d_cnt][R][input_size] with R = 2^branch_depth replica
// rows per (neuron, dendrite). Segment s (heap-indexed, root 1, level
// l = bit_width(s) - 1) owns the rows I(s) = [s * 2^(L-l) - R, (s+1) * 2^(L-l) - R)
// of every input column it covers. Per cell:
//
//   L1 values       permanence in [0, 1]; adapt_history finite and >= 0 (it
//                   accumulates theta, clamped to [0, 1]); mismatch finite (its
//                   sign rests on the unvalidated mismatch_smoothing)
//
// and per (neuron, dendrite, input) column:
//
//   L2 range        1 <= seg <= 2^(L+1) - 1
//   L3 alignment    a row holding seg s lies in I(s)
//   L4 full runs    a column holding s in one row of I(s) holds it in all of them
//   L5 coherence    the rows of one run are copies of one logical synapse:
//                   permanence, mismatch and adapt_history bitwise equal
//
// The compute relies on all four: the response reuses row r-1's penalty draw
// when the segment id repeats (L3-L5), and the split rewrites a whole run from
// its first row, writing into the next dendrite if that row is not a run start
// (L3). Learning keeps them (adapt is a function of segment, input and the old
// permanence; a split writes one value per child run), and the initial state
// holds them, so only a loaded state can break them.
//
// Returns true iff every cell satisfies L1 and every column L2-L5. On false, *why (if non-null)
// names the first violating column, the row and the law. The spans must all
// have neuron_cnt * d_cnt * 2^branch_depth * input_size elements (the caller
// checks sizes; a mismatch returns false).
[[nodiscard]] bool check_bank_layout(std::span<const float>        permanences,
                                     std::span<const std::uint8_t> segment_ids,
                                     std::span<const float>        adapt_history,
                                     std::span<const float>        mismatch,
                                     std::size_t                   neuron_cnt,
                                     std::size_t                   d_cnt,
                                     std::uint8_t                  branch_depth,
                                     std::size_t                   input_size,
                                     std::string                  *why);

} // namespace ngm3

#endif // HD_NGM3_LAYOUT_H
