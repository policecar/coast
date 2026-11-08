//
// Shared per-sample index sequencer for dataset loaders. See sample_sequencer.h.
//

#include "io/sample_sequencer.h"

#include "logging/state_io.h"
#include "logging/rng_state_io.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace dataset {

namespace {

std::string trim(std::string s)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_space).base(), s.end());
    return s;
}

std::size_t parse_index_token(const std::string& token, const std::string& path,
                              std::size_t line_no)
{
    try {
        if (token.empty() || token.front() == '-') {
            throw std::invalid_argument("expected a non-negative integer");
        }
        std::size_t consumed = 0;
        const unsigned long long raw = std::stoull(token, &consumed, 10);
        if (consumed != token.size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (raw > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            throw std::out_of_range("index too large");
        }
        return static_cast<std::size_t>(raw);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid index token '" + token + "' in index_manifest '" +
                                 path + "' at line " + std::to_string(line_no) +
                                 ": " + e.what());
    }
}

void validate_index_order(const std::vector<std::size_t>& indices,
                          std::size_t source_count,
                          const std::string& label)
{
    for (const std::size_t idx : indices) {
        if (idx >= source_count) {
            throw std::runtime_error("Index " + std::to_string(idx) + " in " + label +
                                     " is out of range for source_count " +
                                     std::to_string(source_count));
        }
    }

    auto sorted = indices;
    std::ranges::sort(sorted);
    const auto dup = std::ranges::adjacent_find(sorted);
    if (dup != sorted.end()) {
        throw std::runtime_error("Duplicate index " + std::to_string(*dup) +
                                 " in " + label);
    }
}

// A saved seq_perm must be a permutation of [0, n): every entry in range and
// none twice. A same-length seq_perm cannot legitimately be anything else
// (save_state writes the realised permutation), so a failure is a corrupt
// checkpoint. O(n) time, n bits of scratch.
void validate_saved_permutation(const std::vector<std::uint64_t>& perm, std::size_t n)
{
    std::vector<bool> seen(n, false);
    for (std::size_t i = 0; i < perm.size(); ++i) {
        const std::uint64_t v = perm[i];
        const char* what = v >= n ? "is out of range" : (seen[v] ? "appears twice" : nullptr);
        if (what != nullptr) {
            throw std::runtime_error(
                "Checkpoint seq_perm[" + std::to_string(i) + "] = " + std::to_string(v) + " " +
                what + ": seq_perm is not a permutation of the " + std::to_string(n) +
                " samples, so the checkpoint is corrupt.");
        }
        seen[v] = true;
    }
}

} // namespace

void ensure_manifest_readable(const std::string& path)
{
    if (path.empty()) {
        return;
    }
    const std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open dataset index_manifest: " + path);
    }
}

std::vector<std::size_t> load_index_manifest(const std::string& path,
                                             std::size_t source_count)
{
    if (path.empty()) {
        return {};
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open dataset index_manifest: " + path);
    }

    std::vector<std::size_t> indices;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (const auto comment = line.find('#'); comment != std::string::npos) {
            line.erase(comment);
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        line = trim(std::move(line));
        if (line.empty()) {
            continue;
        }

        std::istringstream is(line);
        std::string token;
        while (is >> token) {
            const std::size_t idx = parse_index_token(token, path, line_no);
            indices.push_back(idx);
        }
    }

    if (indices.empty()) {
        throw std::runtime_error("Dataset index_manifest is empty: " + path);
    }
    validate_index_order(indices, source_count, "index_manifest '" + path + "'");
    return indices;
}

sample_sequencer::sample_sequencer(std::size_t image_count, int change_interval,
                                   bool shuffle, std::uint32_t seed,
                                   std::vector<std::size_t> index_order)
    : image_count_(index_order.empty() ? image_count : index_order.size()),
      change_interval_(change_interval),
      shuffle_(shuffle),
      index_order_(std::move(index_order)),
      rng_(seed)
{
    // Re-validate even though load_index_manifest already checked file-sourced
    // manifests: index_order can also be passed directly (tests, programmatic
    // callers), so the ctor stays the safety net for range and duplicates.
    validate_index_order(index_order_, image_count, "sample_sequencer index_order");
    // SEQUENTIAL never touches rng_ (golden-safety invariant): only build a
    // permutation when shuffling a non-empty set.
    if (shuffle_ && image_count_ > 0) {
        regenerate_permutation();
    }
    refresh_cur_idx();
}

void sample_sequencer::regenerate_permutation()
{
    fill_permutation(perm_, image_count_, rng_);
}

void sample_sequencer::fill_permutation(std::vector<std::size_t>& perm, std::size_t n,
                                        fast_rng_t& rng)
{
    perm.resize(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    // Fisher-Yates from the high end, drawing from the dedicated RNG.
    //
    // Portability note: std::uniform_int_distribution is not specified to yield
    // the same value sequence across standard-library implementations (libstdc++
    // vs libc++), so a *freshly generated* permutation is reproducible only within
    // one toolchain. This does not affect checkpoint resume: save_state persists
    // the realised permutation verbatim (seq_perm), so a resumed run replays the
    // exact order regardless of stdlib -- only future epochs regenerated after the
    // load are stdlib-dependent. coast builds with libstdc++ (GCC) everywhere, so
    // this is a note, not a present bug.
    for (std::size_t i = n; i-- > 1;) {
        std::uniform_int_distribution<std::size_t> d(0, i);
        std::swap(perm[i], perm[d(rng)]);
    }
}

void sample_sequencer::refresh_cur_idx()
{
    cur_idx_ = mapped_index(perm_, pos_);
}

std::size_t sample_sequencer::mapped_index(const std::vector<std::size_t>& perm,
                                           std::size_t pos) const
{
    if (image_count_ == 0) {
        return 0;
    }
    const std::size_t logical_idx = shuffle_ ? perm[pos] : pos;
    return index_order_.empty() ? logical_idx : index_order_[logical_idx];
}

std::size_t sample_sequencer::logical_pos_for_saved_index(std::size_t saved_idx) const
{
    if (index_order_.empty()) {
        return saved_idx;
    }
    const auto it = std::find(index_order_.begin(), index_order_.end(), saved_idx);
    if (it == index_order_.end()) {
        throw std::runtime_error(
            "Legacy checkpoint sample index " + std::to_string(saved_idx) +
            " is not in this sequencer's index_manifest, so the saved cursor has"
            " no position on this subset: a checkpoint continues its sequence only"
            " on the manifest it was saved with (SPEC.md S.5.14).");
    }
    return static_cast<std::size_t>(std::distance(index_order_.begin(), it));
}

void sample_sequencer::advance()
{
    // Legacy behaviour: with change_interval <= 0 the loader emits a static frame
    // and never advances the cursor.
    if (change_interval_ <= 0) {
        return;
    }
    if (++ci_cnt_ < static_cast<std::size_t>(change_interval_)) {
        return;   // still holding the current sample
    }
    ci_cnt_ = 0;
    if (image_count_ == 0) {
        return;
    }
    if (++pos_ >= image_count_) {
        pos_ = 0;
        ++cur_epoch_;
        if (shuffle_) {
            regenerate_permutation();
        }
    }
    refresh_cur_idx();
}

void sample_sequencer::save_state(logging::state_writer& writer) const
{
    // cur_epoch / cur_idx / ci_cnt keep their legacy names so checkpoints written
    // by the pre-sequencer loaders still resume their position; seq_pos / seq_perm
    // / seq_rng are the additional shuffle state.
    writer.write("cur_epoch", static_cast<std::uint64_t>(cur_epoch_));
    writer.write("cur_idx", static_cast<std::uint64_t>(cur_idx_));
    writer.write("ci_cnt", static_cast<std::uint64_t>(ci_cnt_));
    writer.write("seq_pos", static_cast<std::uint64_t>(pos_));
    if (shuffle_) {
        // perm_ holds std::size_t indices; persist them through the fixed-width
        // uint64 array channel (std::size_t is not itself a state_io channel
        // type, and is a *distinct* type from std::uint64_t on non-LP64 targets).
        // The copy is a no-op widening on LP64 and is negligible against the HDF5
        // write it feeds.
        const std::vector<std::uint64_t> perm64(perm_.begin(), perm_.end());
        writer.write("seq_perm", std::span<const std::uint64_t>(perm64));
    }
    logging::write_rng(writer, "seq_rng", rng_);
}

void sample_sequencer::load_state(logging::state_reader& reader)
{
    // Validate, then commit: everything is read and checked into locals, and the
    // members are assigned only once nothing can throw, so a refused load leaves
    // the sequencer as it was. (The env halts on any refused entity load anyway --
    // other entities may already be loaded -- so this keeps the sequencer itself
    // consistent rather than making the env load atomic.)
    const auto cur_epoch = static_cast<std::size_t>(reader.read_uint64("cur_epoch", cur_epoch_));
    const auto ci_cnt    = static_cast<std::size_t>(reader.read_uint64("ci_cnt", ci_cnt_));
    const std::uint64_t saved_cur_idx =
        reader.read_uint64("cur_idx", static_cast<std::uint64_t>(pos_));
    const bool had_seq_pos = reader.has_field("seq_pos");
    // Pre-sequencer checkpoints have no seq_pos; fall back to the legacy cur_idx.
    // With an index_manifest, cur_idx is a source index, so map it back into the
    // manifest's logical position (throws if it is not in the manifest -- a saved
    // cursor has no position on a disjoint subset).
    std::size_t pos = had_seq_pos
        ? static_cast<std::size_t>(reader.read_uint64("seq_pos", saved_cur_idx))
        : logical_pos_for_saved_index(static_cast<std::size_t>(saved_cur_idx));

    fast_rng_t rng = rng_;
    logging::read_rng(reader, "seq_rng", rng, "sample_sequencer");

    std::vector<std::size_t> perm = perm_;
    if (shuffle_) {
        // Read through the fixed-width uint64 channel, then narrow back into the
        // std::size_t perm_. HDF5 widens an older uint32 seq_perm on read, so
        // pre-uint64 checkpoints still load. assign (not move) bridges the
        // size_t/uint64 type distinction on non-LP64 targets portably.
        const auto p = reader.read_uint64_vec("seq_perm");
        if (p.size() == image_count_) {
            // Before any entry is used as an index (#238).
            validate_saved_permutation(p, image_count_);
            perm.assign(p.begin(), p.end());
        } else if (image_count_ > 0) {
            // Cross-dataset load (e.g. eval onto a different set) or a checkpoint
            // written before shuffle existed: rebuild a valid permutation, drawing
            // from the restored RNG as before.
            fill_permutation(perm, image_count_, rng);
        }
    }

    // The saved position reflects the *training* dataset's length; wrap it into
    // the loaded dataset so loading onto a smaller set can't read out of range
    // (mirrors the legacy `cur_idx %= N` guard in every loader's load_state).
    pos = (image_count_ == 0) ? 0 : (pos % image_count_);
    const std::size_t cur_idx = mapped_index(perm, pos);

    // A modern checkpoint's seq_pos is a logical position into whatever manifest
    // was in force when it was saved; loading it under a *different* manifest
    // would silently resume at an unrelated sample. Require the restored source
    // index to match the saved one (the no-seq_pos path is already validated by
    // logical_pos_for_saved_index above).
    if (had_seq_pos && !index_order_.empty() &&
        cur_idx != static_cast<std::size_t>(saved_cur_idx)) {
        throw std::runtime_error(
            "Checkpoint sample cursor (source index " + std::to_string(saved_cur_idx) +
            ") is not where this index_manifest maps, so resuming would start at an"
            " unrelated sample: a checkpoint continues its sequence only on the"
            " manifest it was saved with (SPEC.md S.5.14).");
    }

    cur_epoch_ = cur_epoch;
    ci_cnt_    = ci_cnt;
    pos_       = pos;
    cur_idx_   = cur_idx;
    rng_       = std::move(rng);
    perm_      = std::move(perm);
}

sample_sequencer build_sequencer(const sequence_params_t& params,
                                 std::size_t image_count)
{
    return sample_sequencer(
        image_count,
        static_cast<int>(params.change_interval),
        params.shuffle,
        std::to_underlying(params.random_seed),
        load_index_manifest(params.index_manifest, image_count));
}

void save_sequenced_state(const sequence_params_t& params, logging::state_writer& writer,
                          const sample_sequencer& seq, const fast_rng_t& rng)
{
    seq.save_state(writer);
    logging::write_rng(writer, "rng", rng);
    save_run_binding(writer, params.run);
}

bool restore_sequenced_state(const sequence_params_t& params,
                             logging::state_reader& reader,
                             sample_sequencer& seq, fast_rng_t& rng, std::string_view who)
{
    if (!restores_sequence(reader, params.run, who)) {
        return false;
    }
    // The sequencer owns the cursor and wraps a saved position into the loaded
    // dataset, so loading onto a smaller set stays in range.
    seq.load_state(reader);
    logging::read_rng(reader, "rng", rng, "dataset");
    return true;
}

} // namespace dataset
