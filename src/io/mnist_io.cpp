//
// Grayscale IDX dataset IO entities (MNIST, EMNIST, KMNIST); see mnist_io.h.
//

#include "io/mnist_io.h"
#include "core/require.h"

#include <algorithm>

namespace dataset {

template <typename Tag>
sim::step_result idx_gray_io<Tag>::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);
    auto outp = output_buf->cur_write_buffer();
    COAST_REQUIRE(outp.size() == get_image_size(), core::error_kind::contract_violation);
    if (seq_.change_interval() > 0) {
        auto img  = get_norm_image(seq_.current_index());
        std::ranges::copy(img,outp.begin());
    } else {
        std::ranges::fill(outp,0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen),0.0f,1.0f);
    }

    // Record which sample is presented this step. This does NOT line up with
    // population/out at equal step_index: IO buffers are double-buffered, so a
    // consumer reads this value one step later -- one step per buffered hop on
    // the feedforward path, and recurrent (attractor) groups have no fixed
    // offset at all. cur_idx is std::size_t (no observe dtype specialisation) -> cast.
    observed_index_ = static_cast<std::uint64_t>(seq_.current_index());  // snapshot before seq_ advances; read in idx_io_base::perform_observations

    seq_.advance();
    return {};
}

template <typename Tag>
std::string idx_gray_io<Tag>::status_str() const
{
    std::string status = std::string(Tag::label) + " IO";
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(seq_.current_epoch());
    status += " | idx: " + std::to_string(seq_.current_index());
    return status;
}

template class idx_gray_io<mnist_tag>;
template class idx_gray_io<emnist_tag>;
template class idx_gray_io<kmnist_tag>;

} // namespace dataset
