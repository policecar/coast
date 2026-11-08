//
// Created by jk on 21.03.26.
// Rewritten 25.05.26: two-bank apical/proximal flat-SoA compute.
//

#ifndef HD_NGM3_NEURON_GROUP_H
#define HD_NGM3_NEURON_GROUP_H

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/entity_element_macros.h"
#include "core/rng/fast_rng.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "ngm/window_tracking.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_host.h"   // bank_inputs_t, routing, validation
#include "inhibition/inhibition.h"
#include "logging/config_record.h"
#include "sim_core/io_entity.h"

namespace ngm3 {

/*
 * ngm3::neuron_group_t — second generation neuron group, redesigned
 * around a flat group-owned SoA layout aimed at SIMD / GPU.
 *
 * Storage is split into two parallel banks (apical, proximal). Both banks
 * share the same shape:
 *   [neuron_cnt × dendrite_cnt × rep_cnt × input_size]
 * where rep_cnt = 2^branch_depth and input_size is the concatenated
 * width of all channels routed to that bank.
 *
 * Bank-shaped private helpers take raw spans so that future translation
 * to GPU device kernels is a 1:1 rewrite per helper.
 *
 * Apical bank is optional (apical_cnt may be 0). Proximal must be >= 1.
 * Output is combined as: act = clamp(apical_max * proximal_max + noise),
 * with apical_max defaulting to 1.0 when the apical bank is absent.
 *
 * The class is a sim::io_entity: input channels routed to either
 * bank are resolved from the channel map in init_io(); the per-step
 * process() override gathers them into concatenated bank buffers
 * before running the bank-shaped compute helpers.
 */
class neuron_group_t : public sim::io_entity {

public:
    ELEMENT_TYPE(neuron_group_t, "ngm3_neuron_group")
    // The group's single inhibition instance is its one child (path
    // "inhibition[0]"), exposing the inhibitor for probing/introspection.
    // Other internal state is surfaced via PROBE() at the group level.
    ELEMENT_CHILD_PTR(inhibitor_, inhibition::inhibitor_t, "inhibition")
    void perform_observations(observe::observe_accessor& acc) override;

    // sim_builder-compatible constructor: looks up a named config in
    // the cfg_store and constructs the group with output_id as its
    // identity.
    explicit neuron_group_t(const cfg_store     &cs,
                            const std::string   &type,
                            sim::output_id_t     output_id,
                            seed_offset_t        seed_offset = {});

    // Resolve input channels against the simulation environment's
    // channel map: per-bank concatenated input widths are computed,
    // inp_ids populated, structure registry initialised. Called once
    // by sim_builder's init-func after construction.
    void init_io(const std::unordered_map<std::string, sim::output_channel_t> &input_map);

    // --- io_entity interface ---
    void set_output_buffer(sim::io_buffer& buf) override;
    void set_input_buffer (sim::output_id_t id, sim::io_buffer& buf) override;

    sim::step_result process() override;

    [[nodiscard]] sim::output_id_t              get_outp_id  () const override;
    [[nodiscard]] std::size_t                   get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t>  get_inp_ids  () const override;
    [[nodiscard]] std::string                   status_str   () const override;

    void save_state(logging::state_writer &writer) const override;
    void load_state(logging::state_reader &reader)       override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override { return effective_seed_; }
    [[nodiscard]] std::optional<seed_t> inhibition_seed() const override
    { return inhibition::random_state_seed(inhibitor_->params()); }

    // Read-only parameter access (for GUI / introspection).
    [[nodiscard]] const neuron_group_params_t& get_params() const { return params; }

    // Runtime tunables (exposed for GUI sliders, like ngm2).
    [[nodiscard]] float            get_common_learning_rate()      const { return common_learning_rate; }
    [[nodiscard]] sigmoid_shape_t  get_weight_filter()             const { return weight_filter; }
    void set_common_learning_rate     (float v) { common_learning_rate      = v; }
    void set_weight_filter            (sigmoid_shape_t v) { weight_filter = v; }

    // ------- structural accessors -------
    [[nodiscard]] std::size_t get_neuron_count()       const { return params.neuron_cnt; }
    [[nodiscard]] std::size_t get_apical_count()       const { return params.apical_cnt; }
    [[nodiscard]] std::size_t get_proximal_count()     const { return params.proximal_cnt; }
    [[nodiscard]] uint8_t     get_branch_depth()       const { return params.max_branch_level; }
    [[nodiscard]] std::size_t get_rep_count()          const { return rep_cnt; }
    [[nodiscard]] uint8_t     get_max_segment_id()     const { return max_seg_id; }
    [[nodiscard]] std::size_t get_apical_input_size()  const { return apical_input_size; }
    [[nodiscard]] std::size_t get_proximal_input_size()const { return proximal_input_size; }

    // ------- representation access -------
    // No-op: the CPU group's storage is already CPU-resident, so the templated
    // visualizers' refresh_vis_mirror() call is a no-op here (the GPU sibling
    // overrides this with an on-demand GPU->CPU readback — IDD §5.7 / S10).
    void refresh_vis_mirror() const {}

    [[nodiscard]] std::span<const float>   get_apical_representation  (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const float>   get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_apical_segment_ids     (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_proximal_segment_ids   (std::size_t n, std::size_t d, std::size_t r) const;

    // ------- aggregate statistics -------
    [[nodiscard]] std::size_t get_total_representation_count() const; // distinct seg_ids across all dendrites of both banks
    [[nodiscard]] float       get_max_mismatch()               const;
    [[nodiscard]] float       get_avg_mismatch()               const;

    // ------- post-inhibition output (size == neuron_cnt) -------
    [[nodiscard]] std::span<const float> get_outp() const { return group_outp; }

protected:
    // This group's configuration record (M.7.8), from its current settings.
    [[nodiscard]] logging::config_record current_config() const;

    // Allocate all bank storage given per-bank concatenated widths.
    // Called from init_io once channel routing has been resolved.
    void init_storage(std::size_t apical_in_size, std::size_t proximal_in_size);

    // Run one processing step over pre-gathered concatenated bank
    // inputs. apical_in may be empty iff params.apical_cnt == 0.
    sim::step_result compute_step(std::span<const float> apical_in,
                                  std::span<const float> proximal_in);

    // Bank-shaped compute helpers. Each takes raw spans + sizes,
    // mirroring the eventual kernel signature.
    struct bank_view_t {
        std::span<float>        permanences;
        std::span<uint8_t>      segment_ids;
        std::span<float>        adapt_history;
        std::span<float>        mismatch;
        std::span<float>        rep_act;       // [n_cnt × d_cnt × rep_cnt]
        std::span<const float>  input_buf;     // [input_size]
        std::span<float>        bank_max;      // [n_cnt]
        std::size_t             d_cnt;
        std::size_t             input_size;
    };

    // ngm3 uses the shared ngm2 input-statistics definition (true data max).
    using input_stats_t = ngm2::input_stats_t;

    static input_stats_t compute_input_stats(std::span<const float> input_buf);

    // Per-step input statistics of one bank, computed PER INPUT CHANNEL as
    // ngm2 does (SPEC.md M.3.2) and expanded to input positions, as
    // ngm25's step context does. With a single channel every value equals the
    // pooled one.
    struct bank_ctx_t {
        // per input position (the position's channel's value)
        std::vector<float> inv_sum;          // 1 / channel sum (0 for a zero-sum channel)
        std::vector<float> half_max;         // channel max / 2: the penalty draw range
        std::vector<float> adapt_atten;      // adapt-side attenuation (ngm2::learning_noise_window)
        std::vector<float> high_thres;       // channel avg / 2 + eps
        std::vector<float> inv_high_range;   // 1 / (1 - high_thres), guarded
        std::vector<float> inv_high_thres;   // 1 / high_thres, guarded
        // whole bank
        float total_sum  = 0.0f;   // sum of the channel sums: the leaf normaliser
        float max_val    = 0.0f;   // max of the channel maxima: the mismatch normaliser
        float resp_atten = 1.0f;   // response attenuation from the min noise and min avg over channels with shape evidence
    };
    static void build_bank_ctx(bank_ctx_t &ctx, const bank_inputs_t &routes,
                               std::span<const float> input_buf);

    void adapt_bank_neuron(const bank_view_t  &v,
                           const bank_ctx_t   &ctx,
                           std::size_t n_idx,
                           float        synapse_weight);

    void branch_bank(const bank_view_t &v);

    // Build a bank_view_t over the apical or proximal storage.
    bank_view_t apical_view  (std::span<const float> input_buf);
    bank_view_t proximal_view(std::span<const float> input_buf);

    // ----------------- members -----------------

    const neuron_group_params_t params;

    // derived
    std::size_t rep_cnt              = 0;
    uint8_t     max_seg_id           = 0;
    std::size_t apical_input_size    = 0;
    std::size_t proximal_input_size  = 0;

    // runtime-tunable copies of params defaults (exposed to GUI later)
    float            local_inhibition_strength;
    float            common_learning_rate;
    sigmoid_shape_t  weight_filter;
    float            stochastic_win_thres;

    // Swappable inhibition for this group (one instance; per-group state).
    // compute_step() delegates to inhibitor_->inhibit(neuron_act, group_outp).
    std::unique_ptr<inhibition::inhibitor_t> inhibitor_;

    fast_rng_t              rgen;
    std::vector<fast_rng_t> neuron_rgens;   // size n_cnt; per-neuron RNGs so the
                                            // parallel adapt/response loops have
                                            // independent, deterministic streams
    std::size_t             input_cnt = 0;

    // Effective RNG seed (params.random_seed + seed_offset) stored at construction
    // so the run manifest can surface it without re-deriving coast's composition rule.
    seed_t effective_seed_ = {};

    // io_entity identity + routing
    sim::output_id_t                       id;
    std::vector<sim::output_id_t>          inp_ids;
    sim::io_buffer*                        output_buf = nullptr;

    // Per-bank channel routing populated by init_io and consumed by
    // process(). Each bank holds an ordered list of (id, size,
    // input-buffer-function) — same order as their corresponding
    bank_inputs_t       apical_inputs;
    bank_inputs_t       proximal_inputs;
    std::vector<float>  apical_input_buf;   // concatenation scratch, size apical_input_size
    std::vector<float>  proximal_input_buf; // concatenation scratch, size proximal_input_size
    bank_ctx_t          apical_ctx;         // per-step per-channel statistics (compute_step)
    bank_ctx_t          proximal_ctx;
    std::vector<uint8_t> penalised_buf;     // penalty outcomes (compute_step), size
                                            // neuron_cnt * max(apical, proximal input size)

    // apical bank (sizes 0 if apical_cnt == 0)
    std::vector<float>   apical_permanences;
    std::vector<uint8_t> apical_segment_ids;
    std::vector<float>   apical_adapt_history;
    std::vector<float>   apical_mismatch;
    std::vector<float>   apical_rep_act;

    // proximal bank
    std::vector<float>   proximal_permanences;
    std::vector<uint8_t> proximal_segment_ids;
    std::vector<float>   proximal_adapt_history;
    std::vector<float>   proximal_mismatch;
    std::vector<float>   proximal_rep_act;

    // per-neuron scratch
    std::vector<float> neuron_apical_max;   // size n_cnt; 1.0 when apical_cnt==0
    std::vector<float> neuron_proximal_max; // size n_cnt
    std::vector<float> neuron_act;          // size n_cnt
    std::vector<float> neuron_mx_act;       // size n_cnt (decaying running max)
    ngm::window_tracking::edges_t window_edges;   // tracked learning windows (learning state)

    // post-inhibition output (size n_cnt)
    std::vector<float> group_outp;

    static constexpr uint8_t MAX_SEG_CNT = 32; // covers branch_depth up to 4 (max_seg_id = 31)
};

} // namespace ngm3

#endif // HD_NGM3_NEURON_GROUP_H
