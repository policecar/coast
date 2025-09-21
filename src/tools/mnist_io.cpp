//
// Created by jk on 04.09.25.
//

#include "mnist_io.h"
#include "hd_ngm2_tools.h"

namespace sim {
mnist_io::mnist_io(
    std::size_t _id,
    int _change_interval,
    const std::string &img_file,
    const std::string &label_file,
    int rnd_seed
) :
    io_entity(),
    mnist_db(img_file,label_file),
    id(_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(_change_interval),
    rgen(rnd_seed),
    dis(0.0f,0.05f) // 5% noise
{}

void mnist_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = function;
}

void mnist_io::process()
{
    auto outp = output_mem();
    if (change_interval > 0) {
        auto img  = get_norm_image(cur_idx);
        std::ranges::copy(img,outp.begin());
    } else {
        std::ranges::fill(outp,0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen),0.0f,1.0f);
    }

    if ((change_interval > 0) && ((++ci_cnt %= change_interval) == 0)) {
        cur_idx = (cur_idx + 1) % get_image_cnt();
        if (cur_idx == 0)
            ++cur_epoch;
    }
}

std::size_t mnist_io::get_outp_id() const
{
    return id;
}

std::size_t mnist_io::get_outp_size() const
{
    return get_image_size();
}

std::string mnist_io::status_str() const
{
    std::string status { "MNIST IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    return status;
}
} // sim