//
// Shared per-sample index sequencer for dataset loaders.
//

#ifndef COAST_SAMPLE_SEQUENCER_H
#define COAST_SAMPLE_SEQUENCER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/rng/fast_rng.h"
#include "io/sequence_params.h"

namespace logging {
class state_writer;
class state_reader;
}

namespace dataset {

class sample_sequencer;

[[nodiscard]] std::vector<std::size_t> load_index_manifest(const std::string& path,
                                                           std::size_t source_count);
// Fail fast on an unreadable manifest path without parsing it: two-phase loaders
// (CIFAR, dSprites) call this before their expensive bulk read. No-op for an
// empty path.
void ensure_manifest_readable(const std::string& path);

// Construct a loader's sequencer from its params against the real image_count:
// load any index_manifest and let the ctor validate it. Two-phase loaders call
// this once image_count is known -- not with a 0 count, which would reject every
// manifest index.
[[nodiscard]] sample_sequencer build_sequencer(const sequence_params_t& params,
                                               std::size_t image_count);

// Save a loader's per-sample cursor, input-noise RNG and identity record
// (dataset_run.h): the counterpart of restore_sequenced_state.
void save_sequenced_state(const sequence_params_t& params, logging::state_writer& writer,
                          const sample_sequencer& seq, const fast_rng_t& rng);

// Restore a loader's per-sample cursor + input-noise RNG from a checkpoint, but
// only when the load continues the run that wrote it (dataset::restores_sequence,
// which also checks the identity record; `who` names the loader in its warning).
// Returns true when state was restored.
bool restore_sequenced_state(const sequence_params_t& params,
                             logging::state_reader& reader,
                             sample_sequencer& seq, fast_rng_t& rng, std::string_view who);

// Encapsulates the indexed-sample schedule every index-based loader used to
// reimplement: present sample `current_index()` for `change_interval` steps,
// then step to the next sample, wrapping at the end of an epoch
// (`current_epoch()` bumps). Two modes:
//
//   - SEQUENTIAL (default): indices run 0, 1, ... N-1 in order each epoch.
//   - SHUFFLE:              each epoch presents a fresh per-epoch Fisher-Yates
//                           permutation of [0, N).
//
// There is intentionally NO with-replacement mode.
//
// @decision DEC-001: golden-safety invariant.
// @rationale In SEQUENTIAL mode the sequencer draws ZERO values from its RNG and
// reproduces the exact legacy advancement (cur_idx 0..N-1, wrap -> epoch++) byte
// for byte. The per-dataset `shuffle` default is now ON, so golden fixtures that
// must stay byte-stable pin `shuffle: false` explicitly (the ngm3 A-tier/GPU
// configs); the mnist seed baselines opted into shuffle and were re-baselined
// against the shuffled order. The sequencer's RNG is dedicated and independent of
// the loaders' input-noise generator, so even in SHUFFLE mode the pixel-noise
// stream is untouched.
class sample_sequencer {
public:
    // `seed` is the loader's random_seed (the seed-offset is already folded in by
    // sim_builder before construction, so distinct seeds yield distinct orders).
    sample_sequencer(std::size_t image_count, int change_interval,
                     bool shuffle, std::uint32_t seed,
                     std::vector<std::size_t> index_order = {});

    // The sample index to present this step.
    [[nodiscard]] std::size_t current_index() const { return cur_idx_; }
    [[nodiscard]] std::size_t current_epoch() const { return cur_epoch_; }
    [[nodiscard]] int         change_interval() const { return change_interval_; }
    [[nodiscard]] bool        is_shuffle() const { return shuffle_; }

    // True on the first step of the current sample's change_interval window --
    // the step on which a new sample just started being shown (it is then held
    // for change_interval steps total). Loaders that need a once-per-sample side
    // effect -- e.g. color_mnist drawing a colour per digit rather than re-rolling
    // it every step -- gate on this. Mirrors the legacy `ci_cnt == 0` check those
    // loaders used before the hold counter moved into the sequencer.
    [[nodiscard]] bool        is_interval_start() const { return ci_cnt_ == 0; }

    // Per-step bookkeeping. Bumps the hold counter; once a sample has been held
    // for `change_interval` steps, steps to the next sample, wrapping at epoch
    // end (and, in SHUFFLE mode, regenerating the permutation for the new epoch).
    // No-op when change_interval <= 0 (the legacy "static frame" behaviour).
    void advance();

    void save_state(logging::state_writer& writer) const;
    void load_state(logging::state_reader& reader);

private:
    void regenerate_permutation();   // Fisher-Yates over [0, image_count_)
    void refresh_cur_idx();          // cur_idx_ = mapped_index(perm_, pos_)
    // Fisher-Yates permutation of [0, n) into `perm`, drawing from `rng`.
    static void fill_permutation(std::vector<std::size_t>& perm, std::size_t n,
                                 fast_rng_t& rng);
    // Source index presented at logical position `pos` under permutation `perm`
    // (perm is ignored in SEQUENTIAL). Requires pos < image_count_ and, in
    // SHUFFLE, perm a permutation of [0, image_count_).
    [[nodiscard]] std::size_t mapped_index(const std::vector<std::size_t>& perm,
                                           std::size_t pos) const;
    // Map a legacy checkpoint's source index to this sequencer's logical
    // position. Throws when a manifest is active and does not contain
    // saved_idx (the saved cursor has no position on a disjoint subset).
    [[nodiscard]] std::size_t logical_pos_for_saved_index(std::size_t saved_idx) const;

    std::size_t image_count_;
    int         change_interval_;
    bool        shuffle_;

    std::size_t ci_cnt_    = 0;   // steps the current sample has been held
    std::size_t pos_       = 0;   // position within the current epoch, [0, N)
    std::size_t cur_idx_   = 0;   // presented sample index this step
    std::size_t cur_epoch_ = 0;

    // Epoch permutation (SHUFFLE only; empty in SEQUENTIAL). std::size_t indices,
    // matching the loaders' image indexing; persisted through the state_io
    // uint64-vector channel. Every dataset still fits in uint32 (largest is
    // dSprites at 737280 < 2^32), so checkpoints written by the earlier uint32
    // channel keep loading -- HDF5 widens uint32 -> uint64 on read.
    std::vector<std::size_t> perm_;
    std::vector<std::size_t> index_order_;

    fast_rng_t rng_;   // dedicated; independent of the loaders' input-noise rgen
};

} // namespace dataset

#endif // COAST_SAMPLE_SEQUENCER_H
