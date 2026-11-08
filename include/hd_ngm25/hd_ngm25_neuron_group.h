//
// Created 09.07.26 — ngm2.5 scaffold: the io_entity group.
// I6 (2026-07-10): fleshed out — process() assembles map §0 (R1..R8), the
// kernels live in hd_ngm25_{response,adapt,branch}.{h,cpp} as ngm25::detail
// free functions and this class is the thin caller. Semantic reference is ngm2,
// transcribed via design/ngm2_5_transcription_map.md (regions R1-R9).
//

#ifndef HD_NGM25_NEURON_GROUP_H
#define HD_NGM25_NEURON_GROUP_H

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/entity_element_macros.h"
#include "core/rng/fast_rng.h"
#include "ngm/window_tracking.h"
#include "hd_ngm25/hd_ngm25_adapt.h"
#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_branch.h"
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_host.h"
#include "hd_ngm25/hd_ngm25_response.h"
#include "inhibition/inhibition.h"
#include "logging/config_record.h"
#include "sim_core/io_entity.h"

namespace ngm25 {

class neuron_group_t : public sim::io_entity {

public:
    ELEMENT_TYPE(neuron_group_t, "ngm25_neuron_group")
    // Like ngm3: the flat model has no neuron/dendrite entity children; the
    // inhibitor is the single child.
    ELEMENT_CHILD_PTR(inhibitor_, inhibition::inhibitor_t, "inhibition")
    void perform_observations(observe::observe_accessor& acc) override;

    explicit neuron_group_t(const cfg_store   &cs,
                            const std::string &type,
                            sim::output_id_t   output_id,
                            seed_offset_t      seed_offset = {});

    // Resolves channel routing (apical channels first, then proximal — one
    // contiguous position range per type, group_shape_t contract), sizes the
    // arena via init-time Poisson permanences (D5), builds the structure
    // registry. Called once by sim_builder after construction. Leaves
    // check_arena_consistency() true.
    void init_io(const std::unordered_map<std::string, sim::output_channel_t> &input_map);

    // --- io_entity interface ---
    void set_output_buffer(sim::io_buffer& buf) override;
    void set_input_buffer (sim::output_id_t id, sim::io_buffer& buf) override;

    // One step, map §0 order: gather -> R1 stats -> R2/R3 response+combine ->
    // running-max update -> R4 inhibition -> [frozen: stop] -> R5 winner
    // (serial primary) + secondary adapt (parallel) -> R8 branch gate. The
    // penalty-policy dispatch happens ONCE here per step (switch on
    // params.penalty_policy), never in the hot loops.
    sim::step_result process() override;

    [[nodiscard]] sim::output_id_t             get_outp_id  () const override;
    [[nodiscard]] std::size_t                  get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t> get_inp_ids  () const override;
    [[nodiscard]] std::string                  status_str   () const override;

    void save_state(logging::state_writer &writer) const override;
    void load_state(logging::state_reader &reader)       override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override
    { return effective_seed_; }
    [[nodiscard]] std::optional<seed_t> inhibition_seed() const override
    { return inhibition::random_state_seed(inhibitor_->params()); }

    [[nodiscard]] const neuron_group_params_t& get_params() const { return params; }

    [[nodiscard]] std::size_t get_neuron_count()  const { return params.neuron_cnt; }
    [[nodiscard]] std::size_t get_synapse_count() const { return arena_.syn_count(); }
    // Snapshots of the learned arrays (for CPU<->GPU adapt validation).
    [[nodiscard]] std::vector<float> perm_snapshot()       const { return arena_.perm; }
    [[nodiscard]] std::vector<float> adapt_hist_snapshot() const { return arena_.adapt_hist; }
    [[nodiscard]] std::vector<float> mismatch_snapshot()   const { return arena_.mismatch; }

    // ------- vis / introspection surface (map R9, pitfall P5) -------
    // Duck-typed to ngm3's contract so ngm2.5 reuses the ngm3 flat/tree
    // renderers unchanged. The sparse CSR arena is materialized into a dense
    // [neuron × dendrite × rep × input_size] mirror per bank (ngm3's layout) on
    // demand by refresh_vis_mirror — the renderers then read it via the span
    // accessors. rep_cnt = 2^branch_depth leaf slots; rep row r is the synapse
    // (permanence + seg id) covering leaf r for each input position.
    [[nodiscard]] std::size_t get_apical_count()   const { return params.apical_cnt; }
    [[nodiscard]] std::size_t get_proximal_count() const { return params.proximal_cnt; }
    [[nodiscard]] uint8_t     get_branch_depth()   const { return params.max_branch_level; }
    [[nodiscard]] std::size_t get_rep_count()      const
    { return std::size_t{1} << params.max_branch_level; }
    [[nodiscard]] uint8_t     get_max_segment_id() const { return shape_.max_seg_id; }
    void refresh_vis_mirror() const;

    [[nodiscard]] std::span<const float>   get_apical_representation  (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const float>   get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_apical_segment_ids     (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_proximal_segment_ids   (std::size_t n, std::size_t d, std::size_t r) const;

    [[nodiscard]] std::size_t get_total_representation_count() const;
    [[nodiscard]] float       get_max_mismatch() const;
    [[nodiscard]] float       get_avg_mismatch() const;
    [[nodiscard]] std::span<const float> get_outp() const { return group_outp_; }

private:
    // Test-only seam: a unit test reaches the private arena to inject a fault
    // mid-life (a repack-preserved corruption) and drive it through process(), so
    // the in-step post-repack consistency guard's halt path is exercised end-to-end
    // — not just via direct checker calls or the load path. Not used in production;
    // see tests/unit/hd_ngm25/test_process_guard.cpp.
    friend struct neuron_group_test_access;

    // R2/R3 response+combine, dispatched on the penalty policy (one runtime
    // branch per step, never per synapse).
    void response_dispatch();

    // ----- members -----
    const neuron_group_params_t params;
    group_shape_t               shape_;
    synapse_arena_t             arena_;
    step_ctx_t                  ctx_;              // rebuilt each step (R1)

    adapt_params_t  adapt_params_;                 // derived from params (R5/R6/R7)
    branch_params_t branch_params_;                // derived from params (R8)

    // per-neuron carried state (persisted, map §1)
    std::vector<std::uint64_t> input_count_;       // per adapt CALL (R8 cadence)
    std::vector<float>         neuron_mx_act_;      // decaying running max
    ngm::window_tracking::edges_t window_edges_;   // tracked learning windows (learning state)

    // per-step scratch (not persisted)
    std::vector<float> dendrite_act_;              // [dend_cnt]
    std::vector<float> neuron_act_;                // [neuron_cnt]
    std::vector<float> group_outp_;                // [neuron_cnt] post-inhibition
    std::vector<float> seg_activity_;              // [dend_cnt x seg_cnt]
    std::vector<float> seg_weights_;               // [dend_cnt x seg_cnt]

    // Dense vis mirror, materialized on demand (mutable: filled by the const
    // refresh_vis_mirror, read by the const span accessors).
    mutable dense_mirror_t mirror_;

    // identity / reproducibility
    sim::output_id_t id_;
    seed_t           effective_seed_ = {};         // random_seed + seed_offset
    std::uint32_t    step_           = 0;          // feeds ALL hash keys; persisted
    std::uint32_t    branch_epoch_   = 0;          // feeds D4 keys; persisted

    // io routing (route_inputs, hd_ngm25_host.h).
    std::vector<channel_route_t>  routes_;
    std::vector<std::size_t>      inp_ids_;        // flat list for get_inp_ids
    sim::io_buffer*               output_buf_ = nullptr;
    std::vector<float>            input_concat_;   // gathered per step

    std::unique_ptr<inhibition::inhibitor_t> inhibitor_;
};

} // namespace ngm25

#endif // HD_NGM25_NEURON_GROUP_H
