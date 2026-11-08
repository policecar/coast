//
// Created by jk on 25.01.26.
//

#ifndef DATASET_JSON_H
#define DATASET_JSON_H

#include <nlohmann/json.hpp>
#include "io/dataset_cfg.h"
#include "diag/diag.h"

#include "io/mnist_cfg.h"
#include "io/dsprites_cfg.h"
#include "io/emnist_cfg.h"
#include "io/kmnist_cfg.h"
#include "io/cifar10_cfg.h"
#include "io/cifar100_cfg.h"
#include "io/chairs3d_cfg.h"
#include "io/color_mnist_cfg.h"
#include "io/sequence_params.h"

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Dataset param serialisers.
//
// The four sequencing fields shared by every sequenced loader live in
// dataset::sequence_params_t, whose serialiser is generated once below and reused by
// each family via NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT — so a new
// cross-cutting field is added in one place. The derived macro emits base fields
// before family fields, changing key order vs. the pre-refactor layout; that is safe
// because dataset config JSON is parsed order-independently and nothing compares
// serialised text, only values. The WITH_DEFAULT variants keep readers null-tolerant
// (a JSON `null` block defaults all fields instead of throwing).
//
// dsprites is hand-written (see below); chairs3d is flat (no sequencer).
//
// NOTE on -Wmaybe-uninitialized: the WITH_DEFAULT macros (and dsprites' hand-written
// from_json) build a throwaway `Type{}` default whose base std::string members go
// unread, which used to trip a GCC -O3 false positive. That is fixed once, at the
// source, by the non-inlinable default constructor on dataset::sequence_params_t
// (see sequence_params.h) — nothing to work around here. Do not reintroduce per-call
// suppressions or hand-write these serialisers for that reason.
// ---------------------------------------------------------------------------

namespace dataset {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(sequence_params_t,
    change_interval,
    noise_range,
    random_seed,
    shuffle,
    index_manifest
)

}

namespace mnist {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(mnist_param_t, dataset::sequence_params_t,
    image_file,
    label_file
)

}

namespace emnist {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(emnist_param_t, dataset::sequence_params_t,
    image_file,
    label_file
)

}

namespace kmnist {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(kmnist_param_t, dataset::sequence_params_t,
    image_file,
    label_file
)

}

namespace color_mnist {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(color_mnist_param_t, dataset::sequence_params_t,
    image_file,
    label_file
)

}

namespace cifar10 {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar10_param_t, dataset::sequence_params_t,
    data_dir,
    use_test_set
)

}

namespace cifar100 {

NLOHMANN_DEFINE_DERIVED_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar100_param_t, dataset::sequence_params_t,
    data_dir,
    use_test_set,
    use_coarse_labels
)

}

namespace dsprites {

// Hand-written (not a derived macro) solely to keep the randomize_readout
// deprecation warning. The shared sequencing fields still delegate to the base
// serialiser, so the field list stays defined in one place. Templated on
// BasicJsonType so it serialises through both nlohmann::json and ordered_json.
// The old `randomize_readout` flag (with-replacement i.i.d. readout) was replaced
// by `shuffle` (per-epoch without-replacement reshuffle); it is now ignored.
template<typename BasicJsonType,
         nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void to_json(BasicJsonType& j, const dsprites_param_t& t) {
    nlohmann::to_json(j, static_cast<const dataset::sequence_params_t&>(t));
    j["hdf5_file"] = t.hdf5_file;
}

template<typename BasicJsonType,
         nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void from_json(const BasicJsonType& j, dsprites_param_t& t) {
    if (j.is_null()) return;   // null value → leave t at its defaults
    nlohmann::from_json(j, static_cast<dataset::sequence_params_t&>(t));
    t.hdf5_file = j.value("hdf5_file", dsprites_param_t{}.hdf5_file);

    if (j.contains("randomize_readout")) {
        COAST_WARN("config", "dsprites config key 'randomize_readout' is deprecated "
                             "and ignored. Use 'shuffle' for per-epoch reshuffling "
                             "(without replacement); when 'shuffle' is omitted the "
                             "default is true.");
    }
}

} // namespace dsprites

namespace chairs3d {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(chairs3d_param_t,
    data_dir,
    image_width,
    image_height,
    change_interval,
    noise_range,
    random_seed
)

}

namespace dataset {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    mnist,
    dsprites,
    emnist,
    kmnist,
    cifar10,
    cifar100,
    chairs3d,
    color_mnist
)


}

#endif //DATASET_JSON_H
