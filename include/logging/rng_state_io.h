//
// Serialization helpers for fast_rng_t (mt19937_simd) generator state.
//
// Persisting the PRNG state lets a checkpointed run reproduce the *exact*
// random stream after a --load-state, so a resumed simulation is bit-identical
// to one that ran straight through. The state is written verbatim under a named
// sub-group of the entity's state record.
//
// Backward/forward compatible: read_rng() leaves the generator at its freshly
// seeded state if the group is absent (a checkpoint written before PRNG
// persistence existed), and older binaries simply ignore the extra group. No
// schema-version gate is required. read_rng() returns whether the group was
// present, so an entity whose format has always carried it can refuse a
// checkpoint without it (see state_checks.h).
//
// A group that is present but incomplete is a corrupt checkpoint for every
// entity: mt, tempered and mti have been written together since the group
// exists (98122325), so no older format produces a partial one. read_rng()
// throws "<who>: corrupt checkpoint — rng <name>: <what>" for it rather than
// keep the seeded stream, which would restart the noise while the load still
// presents itself as an exact resume.
//

#ifndef COAST_RNG_STATE_IO_H
#define COAST_RNG_STATE_IO_H

#include "logging/state_io.h"
#include "core/rng/fast_rng.h"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace logging {

inline void write_rng(state_writer& w, std::string_view name, const fast_rng_t& rng)
{
    const auto& g = rng.generator();
    w.begin_group(name);
    w.write("mt",       g.state_words());
    w.write("tempered", g.tempered_words());
    w.write("mti",      static_cast<std::uint64_t>(g.position()));
    w.end_group();
}

inline bool read_rng(state_reader& r, std::string_view name, fast_rng_t& rng, std::string_view who)
{
    if (!r.enter_group(name)) return false;  // pre-PRNG checkpoint: keep seeded stream
    constexpr std::size_t n = core::rng::mt19937_simd::state_size;
    auto mt  = r.read_uint32_vec("mt");
    auto tmp = r.read_uint32_vec("tempered");
    // mti is the read position inside the tempered batch: without it (or with
    // one past the batch) the restored stream would not continue where it stopped.
    const bool        has_mti = r.has_field("mti");
    const std::size_t mti     = has_mti ? static_cast<std::size_t>(r.read_uint64("mti")) : 0;
    r.leave_group();

    std::string what;
    auto note = [&](std::string s) { what += what.empty() ? s : ", " + s; };
    if (mt.size() != n)  note(std::format("mt has {} words, expected {}", mt.size(), n));
    if (tmp.size() != n) note(std::format("tempered has {} words, expected {}", tmp.size(), n));
    if (!has_mti)        note("mti missing");
    else if (mti > n)    note(std::format("mti {} is past the batch of {}", mti, n));
    if (!what.empty())
        throw std::runtime_error(std::format("{}: corrupt checkpoint — rng {}: {}", who, name, what));

    rng.generator().restore_state(mt, tmp, mti);
    return true;
}

} // namespace logging

#endif // COAST_RNG_STATE_IO_H
