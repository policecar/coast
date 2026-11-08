//
// Created for the COAST swappable-inhibition feature.
//
// random_index: random recurrent lateral inhibition. Each neuron group owns an
// N×N permanence matrix M (entries U[0,1), zero diagonal) generated once from a
// seed. A variant rule derives a fixed weight matrix T from M; each step the
// inhibitor computes inh = T·raw and out = clamp(raw − inh, 0, 1). The intent is
// a simpler, more biologically-grounded inhibition — random
// inhibitory connections, no modelled interneurons — with two knobs: the breadth
// of inhibition (reach) and its strength (strength).
//
// CPU only this cycle (make_gpu_inhibitor rejects this kind). M is persisted
// directly in the simulation state, so save/load is faithful on CPU without
// depending on RNG-stream restoration.
//

#ifndef COAST_INHIBITION_RANDOM_INDEX_H
#define COAST_INHIBITION_RANDOM_INDEX_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/entity_element_macros.h"
#include "inhibition/inhibition.h"
#include "inhibition/inhibition_cfg.h"

namespace inhibition {

class random_index_t : public inhibitor_t {
public:
    explicit random_index_t(const inhibition_params_t& p);

    ELEMENT_TYPE(random_index_t, "random_index")

    // raw and out may alias (every out[i] is written explicitly from raw[i] and
    // the precomputed inhibition, so the transform is alias-safe in place).
    // Precondition: raw.size() == out.size(). The matrix is lazily generated on
    // the first call, taking N from raw.size().
    void inhibit(std::span<const float> raw, std::span<float> out) override;

    // Full state: variant, reach, strength, seed, n, and the matrix M. The set is
    // complete so a checkpoint is self-describing. On load the live config
    // hyperparameters take precedence (T is rebuilt from the restored M against
    // them); variant, reach and strength are written for forensics/provenance but
    // not read back. The seed is read back with M — it is random state, the
    // provenance of the restored matrix. A matrix whose n is not `neuron_count`,
    // or whose `n` field disagrees with its shape, is refused before anything
    // is assigned.
    void save_state(logging::state_writer& w) const override;
    void load_state(logging::state_reader& r, std::size_t neuron_count) override;

    // Inspection seam (used by unit tests and available for probing). Empty until
    // the matrix is generated on the first inhibit() / restored on load.
    [[nodiscard]] std::span<const float> permanence() const { return matrix; }
    [[nodiscard]] std::span<const float> weights()    const { return weight; }
    [[nodiscard]] std::size_t            size()       const { return n; }

private:
    std::size_t        n = 0;
    std::vector<float> matrix;   // n·n permanence M, row-major, U[0,1), zero diagonal
    std::vector<float> weight;   // n·n derived weights T
    std::vector<float> inh;      // n scratch for inh = T·raw (no per-step allocation)
    bool               initialized = false;

    // Allocate and generate M (from seed) + derive T. Idempotent per instance.
    void ensure_init(std::size_t neuron_count);

    // Derive T from M for the configured variant. Isolated so a future modulatory
    // input can re-derive T after changing a hyperparameter (none wired yet).
    void recompute();
};

}

#endif // COAST_INHIBITION_RANDOM_INDEX_H
