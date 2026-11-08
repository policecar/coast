//
// Shared sequencing parameters for index-based dataset loaders.
// All seven sequenced loaders (mnist, emnist, kmnist, color_mnist, dsprites,
// cifar10, cifar100) inherit publicly from this base struct so that
// field access at use sites remains `params.change_interval` etc.
//
// chairs3d is deliberately excluded: it has no shuffle and no sample_sequencer.
//

#ifndef COAST_SEQUENCE_PARAMS_H
#define COAST_SEQUENCE_PARAMS_H

#include <cstddef>
#include <string>
#include <utility>

#include "core/seed.h"
#include "io/dataset_run.h"

namespace dataset {

struct sequence_params_t {
    // Non-inlinable, user-provided default constructor. This exists solely to defeat a
    // GCC -O3 false-positive -Wmaybe-uninitialized: this struct has a std::string
    // (index_manifest) with an NSDMI, and any site that default-constructs a value of
    // this type (or a derived *_param_t) and lets it die without reading index_manifest
    // — the JSON serialisers' fallback `Type{}`, the GUI's `map[name] = {}`, etc. — gets
    // its string construction scalarized away by GCC's SRA pass, which then warns on the
    // string destructor (basic_string::_M_dispose). The NSDMIs always run, so it is a
    // genuine false positive, but it is inlining-dependent and roams between call sites
    // and TUs, so there is no single call-site to suppress. Routing every default
    // construction through one out-of-line ctor makes the initialisation opaque to the
    // optimizer at those sites, so SRA has nothing to scalarize and the FP disappears
    // everywhere. [[gnu::noinline]] is the load-bearing part (an inlinable `= default`
    // would let the FP back in); clang accepts the attribute and never had the FP.
    // The tiny per-construction call cost is irrelevant — these configs are built only
    // on config load / GUI edits. Do NOT remove without re-checking a clean GCC build.
    [[gnu::noinline]] sequence_params_t() {}

    std::size_t              change_interval = 15;
    std::pair<double,double> noise_range     = {0.0, 0.05};
    seed_t                   random_seed     = {};
    bool                     shuffle         = true;

    // Path to an index manifest: a text file of sample indices into the loaded
    // dataset. When set, the loader presents exactly those samples — in
    // manifest order when shuffle is false, reshuffled per epoch within the
    // manifest when true. Empty (the default) presents the full dataset.
    std::string index_manifest = "";

    // The loader's identity record, checked when a checkpoint load restores the
    // sequence (dataset_run.h); whether it restores is the run's (run_id). Not a
    // configured setting: sim::register_entity_handlers sets it.
    run_binding_t run;
};

} // namespace dataset

#endif // COAST_SEQUENCE_PARAMS_H
