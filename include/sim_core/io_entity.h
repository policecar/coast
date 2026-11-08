//
// Created by jk on 03.09.25.
//

#ifndef SIM_IO_ENTITY_H
#define SIM_IO_ENTITY_H

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include "sim_core/io_buffer.h"
#include "core/entity_element.h"
#include "core/seed.h"
#include "core/step_error.h"
#include "core/structure_registry.h"
#include "probe/entity_probe_set.h"

namespace logging {
class state_writer;
class state_reader;
}

namespace sim {

using output_id_t = std::size_t;
using output_channel_t = std::pair<output_id_t,std::size_t>;

/// Result of a per-step process() call: success, or a hot-path core::step_error.
/// process() reports errors by value through this channel (not by throwing), so a
/// violated invariant surfaces to the controller as a clean halt rather than an
/// exception unwinding out of a TBB worker. See design/error_handling_plan.md.
using step_result = std::expected<void, core::step_error>;

/// Bridge from the value-based step channel back to an exception, for cold-path
/// callers that are NOT the sim controller: probe/benchmark harnesses, tools, and
/// tests that drive process()/swap_io() directly and have no fault sink. The
/// controller consumes step_result by value (see sim_thread_controller); these
/// callers instead want a faulted step to abort loudly with a diagnostic. Routing
/// the result through here also consumes it, satisfying the [[nodiscard]] that
/// std::expected carries (GCC warns on a dropped step_result). No-op on success.
inline void throw_on_step_fault(const step_result& r, const char* what) {
    if (!r) {
        const core::step_error& e = r.error();
        throw std::runtime_error(std::string(what) + " faulted: " +
                                 core::to_string(e.kind) + " @ " + e.site);
    }
}

struct io_entity : public core::entity_element {

    io_entity() = default;
    ~io_entity() override = default;

    // Non-copyable (structure_registry is non-copyable)
    io_entity(const io_entity&) = delete;
    io_entity& operator=(const io_entity&) = delete;

    // Movable
    io_entity(io_entity&&) = default;
    io_entity& operator=(io_entity&&) = default;

    // Bind the buffer this entity writes its output into.
    virtual void set_output_buffer(io_buffer&) {}
    // Bind one of this entity's input buffers (identified by its source output id).
    virtual void set_input_buffer(output_id_t, io_buffer&) {}

    virtual step_result process() = 0;

    [[nodiscard]] virtual output_id_t get_outp_id() const = 0;
    [[nodiscard]] virtual std::size_t get_outp_size() const = 0;
    [[nodiscard]] virtual std::span<const std::size_t> get_inp_ids() const = 0;

    [[nodiscard]] virtual std::string status_str() const { return ""; }

    /// Effective RNG seed this entity was constructed with — the config seed
    /// with the global (--seed-offset) and per-node offsets already folded in,
    /// as actually used to seed this entity's RNG. Recorded in the run manifest
    /// for provenance so a consumer needn't reconstruct coast's seed composition.
    /// nullopt for entities with no seeded randomness.
    [[nodiscard]] virtual std::optional<seed_t> effective_seed() const { return std::nullopt; }

    /// Seed of the entity's inhibition matrix (a neuron group's random_index
    /// inhibitor), recorded in the run manifest next to effective_seed(): it
    /// composes with the same offsets, but seeds a separate stream, and a drawn
    /// one (ri_seed_from_time) appears nowhere else. nullopt when there is none.
    [[nodiscard]] virtual std::optional<seed_t> inhibition_seed() const { return std::nullopt; }

    // -- State persistence support --

    /// Save entity state for later restoration.
    /// Override in derived classes to save entity-specific state.
    /// Default implementation does nothing.
    virtual void save_state([[maybe_unused]] logging::state_writer& writer) const {}

    /// Load entity state from previously saved data.
    /// Override in derived classes to restore entity-specific state.
    /// Default implementation does nothing.
    virtual void load_state([[maybe_unused]] logging::state_reader& reader) {}

    // -- Entity element hierarchy support --

    /// Access the structure registry for this entity.
    [[nodiscard]] core::structure_registry& structure() { return structure_; }
    [[nodiscard]] const core::structure_registry& structure() const { return structure_; }

    /// Access the probe set for this entity.
    [[nodiscard]] probe::entity_probe_set& probes() { return probes_; }
    [[nodiscard]] const probe::entity_probe_set& probes() const { return probes_; }

    /// Initialize the structure registry and probe set.
    /// Call this after all child elements have been created, and after the entity
    /// is in its final memory location (not subject to vector reallocation).
    void init_structure() {
        structure_.build(this);
        probes_.init(this, &structure_);
    }

protected:
    core::structure_registry structure_;
    probe::entity_probe_set probes_;
};


}

#endif //SIM_IO_ENTITY_H
