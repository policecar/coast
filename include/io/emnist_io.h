//
// EMNIST dataset IO entity
//

#ifndef EMNIST_IO_H
#define EMNIST_IO_H

#include <functional>
#include <span>
#include <string>

#include "core/rng/fast_rng.h"
#include "io_entity.h"
#include "mnist_db.h"
#include "emnist_cfg.h"
#include "core/entity_element_macros.h"

namespace emnist {

class emnist_io : public sim::io_entity, public mnist::mnist_db {
public:
    // Entity element interface
    ELEMENT_TYPE(emnist_io, "emnist_input")

private:
    const emnist_param_t params;

    std::function<std::span<float>()> output_mem {};

    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;

public:
    explicit emnist_io(const emnist_param_t &params, sim::output_id_t output_id);

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] int get_change_interval() { return change_interval; }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { 28, 28, 1 };
    }
    [[nodiscard]] std::size_t sample_count() const {
        return get_image_cnt();
    }
    [[nodiscard]] std::size_t current_epoch() const {
        return cur_epoch;
    }
    [[nodiscard]] std::size_t current_index() const {
        return cur_idx;
    }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        return get_norm_image(idx);
    }
};

} // emnist

#endif //EMNIST_IO_H
