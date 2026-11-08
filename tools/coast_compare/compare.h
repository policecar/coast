#ifndef COAST_COMPARE_COMPARE_H
#define COAST_COMPARE_COMPARE_H

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace coast::compare {

// Tolerance threshold: |a-b| < max(rel*max(|a|,|b|), abs_floor)
struct tolerance {
    double rel       = 1e-6;
    double abs_floor = 1e-9;
};

// Comparison mode. strict = element-wise (the original behaviour). aggregate =
// numeric arrays are reduced to (mean, max, std) and those three statistics are
// compared instead of the elements — for Shape-A statistical-aggregate parity
// where per-element values diverge by design (e.g. CPU vs GPU RNG streams) but
// the spatial distribution of the final-step buffer matches.
enum class compare_mode { strict, aggregate };

// The three per-array statistics aggregate mode compares.
enum class agg_stat { mean, max, stddev };

// Aggregate tolerance for one path: a base applied to all three statistics,
// with optional per-statistic refinements (std usually needs looser bounds).
struct agg_tolerance {
    tolerance                base;
    std::optional<tolerance> mean;
    std::optional<tolerance> max;
    std::optional<tolerance> stddev;
};

// Per-field tolerance overrides. Keys are exact paths
// (e.g. "group_2/spike_counts"); element-index suffixes like "[42]" are
// stripped before lookup.
class tolerance_set {
public:
    tolerance_set() = default;

    // Load from JSON file shaped like:
    //   { "default": {"rel": 1e-6, "abs_floor": 1e-9},
    //     "overrides": { "group_2/spike_counts": {"rel": 1e-4} } }
    // Throws std::runtime_error on parse / structure failure.
    static tolerance_set load(const std::filesystem::path& json_path);

    [[nodiscard]] tolerance get(std::string_view path) const;
    [[nodiscard]] tolerance default_tol() const noexcept { return default_; }
    void set_default(tolerance t) noexcept { default_ = t; }
    void set_override(std::string path, tolerance t);

    // ----- aggregate-mode tolerances -----
    // get_aggregate resolves the tolerance for (path, stat) with precedence:
    //   override[path].<stat>  →  override[path].base
    //     →  aggregate-default.<stat>  →  aggregate-default.base
    // get_aggregate_base is the same chain minus the per-stat steps (used for
    // scalar fields in aggregate mode). When no aggregate block was loaded, the
    // base falls back to the strict default_, so aggregate mode still runs.
    [[nodiscard]] bool      has_aggregate() const noexcept { return has_aggregate_; }
    [[nodiscard]] tolerance get_aggregate(std::string_view path, agg_stat stat) const;
    [[nodiscard]] tolerance get_aggregate_base(std::string_view path) const;
    void set_aggregate_default(agg_tolerance t);
    void set_aggregate_override(std::string path, agg_tolerance t);

private:
    tolerance                       default_;
    std::map<std::string, tolerance> overrides_;

    bool                              has_aggregate_ = false;
    agg_tolerance                     agg_default_{};   // .base seeded from default_ at load
    std::map<std::string, agg_tolerance> agg_overrides_;
};

enum class field_status { pass, fail_tol, fail_structural };

struct field_result {
    field_status status      = field_status::pass;
    std::string  path;
    double       golden_value = 0.0;
    double       new_value    = 0.0;
    double       rel_delta    = 0.0;   // |a-b| / max(|a|,|b|), 0 when both zero
    tolerance    applied_tol;
    std::string  note;                 // structural reason, or extra detail for fail_tol
};

struct golden_match {
    std::filesystem::path     golden_path;
    std::vector<field_result> fields;
    [[nodiscard]] bool        all_pass()       const noexcept;
    [[nodiscard]] bool        any_structural() const noexcept;
    [[nodiscard]] std::size_t failing_count()  const noexcept;
    [[nodiscard]] double      max_rel_delta()  const noexcept;
};

struct compare_outcome {
    enum class result { matched, no_match, structural_error };
    result                              kind = result::structural_error;
    std::optional<golden_match>         matched_golden;   // when kind == matched
    std::optional<golden_match>         closest_golden;   // when kind != matched
    std::vector<std::filesystem::path>  goldens_considered;
    std::string                         note;             // optional context (e.g. "no goldens dir")
};

// Walks <goldens_dir>/*.json chronologically (β naming sorts lexicographically).
// Returns matched on the first all-pass golden; otherwise no_match with the
// closest structurally-clean candidate; otherwise structural_error.
compare_outcome compare_against_goldens(
    const std::filesystem::path& candidate_blob,
    const std::filesystem::path& goldens_dir,
    const tolerance_set&         tolerances,
    compare_mode                 mode = compare_mode::strict);

// Render per IDD §5: header line + flat severity-sorted FAIL list. With
// verbose, PASS entries are appended below.
void render_report(std::ostream& out, const compare_outcome& outcome, bool verbose);

// Exit code derived from outcome.kind: matched=0, no_match=1, structural_error=2.
[[nodiscard]] int exit_code_of(const compare_outcome& outcome) noexcept;

} // namespace coast::compare

#endif // COAST_COMPARE_COMPARE_H
