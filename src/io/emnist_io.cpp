//
// EMNIST dataset IO entity implementation
//

#include "../../include/io/emnist_io.h"

namespace emnist {

emnist_io::emnist_io(const emnist_param_t &params, sim::output_id_t output_id) :
    mnist::mnist_db(params.image_file, params.label_file),
    params(params),
    id(output_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(params.change_interval),
    rgen(params.random_seed),
    dis(params.noise_range.first, params.noise_range.second)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
}

void emnist_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = std::move(function);
}

void emnist_io::process()
{
    if (!output_mem) {
        throw std::runtime_error("emnist_io::process() called before set_outp_func()");
    }
    auto outp = output_mem();
    if (outp.size() != get_image_size()) {
        throw std::runtime_error("Output span size mismatch: expected " +
            std::to_string(get_image_size()) + ", got " + std::to_string(outp.size()));
    }
    if (change_interval > 0) {
        auto img  = get_norm_image(cur_idx);
        std::ranges::copy(img,outp.begin());
    } else {
        std::ranges::fill(outp,0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen),0.0f,1.0f);
    }

    if (change_interval > 0 && ++ci_cnt >= static_cast<std::size_t>(change_interval)) {
        ci_cnt = 0;
        if (++cur_idx >= get_image_cnt()) {
            cur_idx = 0;
            ++cur_epoch;
        }
    }
}

sim::output_id_t emnist_io::get_outp_id() const
{
    return id;
}

std::size_t emnist_io::get_outp_size() const
{
    return get_image_size();
}

std::string emnist_io::status_str() const
{
    std::string status { "EMNIST IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    return status;
}

} // emnist
