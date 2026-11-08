//
// Grayscale IDX dataset IO entities: MNIST, EMNIST, KMNIST.
//
// The three loaders are one implementation (idx_gray_io) that differs only in
// its config type and names. Each stays a distinct C++ type, so its
// element_type, its io_entity_type_t dispatch and its "<name>_input" structural
// path are unchanged. color_mnist (RGB output) keeps its own class.
//

#ifndef MNIST_IO_H
#define MNIST_IO_H

#include <span>
#include <string>
#include <tuple>

#include "io/idx_io_base.h"
#include "io/mnist_cfg.h"
#include "io/emnist_cfg.h"
#include "io/kmnist_cfg.h"
#include "core/entity_element_macros.h"

namespace dataset {

// Tag: config type, element_type display name, status_str() label.
template <typename Param, const char* TypeName, const char* Label>
struct idx_gray_tag {
    using param_t = Param;
    static constexpr const char* type_name = TypeName;
    static constexpr const char* label     = Label;
};

template <typename Tag>
class idx_gray_io final : public idx_io_base {
public:
    ELEMENT_TYPE(idx_gray_io, Tag::type_name)

    explicit idx_gray_io(const typename Tag::param_t& params, sim::output_id_t output_id)
        : idx_io_base(params.image_file, params.label_file, output_id, params) {}

    sim::step_result process() override;

    [[nodiscard]] std::string status_str() const override;

    // dataset_io interface (see gui_core/dataset_access.cpp)
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const { return { 28, 28, 1 }; }
    [[nodiscard]] std::size_t sample_count() const { return get_image_cnt(); }
    [[nodiscard]] std::size_t current_epoch() const { return seq_.current_epoch(); }
    [[nodiscard]] std::size_t current_index() const { return seq_.current_index(); }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        return get_norm_image(idx);
    }
};

inline constexpr char mnist_type_name[]  = "mnist_input";
inline constexpr char emnist_type_name[] = "emnist_input";
inline constexpr char kmnist_type_name[] = "kmnist_input";
inline constexpr char mnist_label[]      = "MNIST";
inline constexpr char emnist_label[]     = "EMNIST";
inline constexpr char kmnist_label[]     = "KMNIST";

using mnist_tag  = idx_gray_tag<mnist::mnist_param_t,   mnist_type_name,  mnist_label>;
using emnist_tag = idx_gray_tag<emnist::emnist_param_t, emnist_type_name, emnist_label>;
using kmnist_tag = idx_gray_tag<kmnist::kmnist_param_t, kmnist_type_name, kmnist_label>;

// Member functions are defined and instantiated in mnist_io.cpp.
extern template class idx_gray_io<mnist_tag>;
extern template class idx_gray_io<emnist_tag>;
extern template class idx_gray_io<kmnist_tag>;

} // namespace dataset

namespace mnist  { using mnist_io  = dataset::idx_gray_io<dataset::mnist_tag>;  }
namespace emnist { using emnist_io = dataset::idx_gray_io<dataset::emnist_tag>; }
namespace kmnist { using kmnist_io = dataset::idx_gray_io<dataset::kmnist_tag>; }

#endif //MNIST_IO_H
