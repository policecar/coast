//
// Created by jk on 04.09.25.
//

#ifndef MNIST_IO_H
#define MNIST_IO_H

#include <functional>
#include <span>
#include <string>
#include <random>

#include "io_entity.h"
#include "mnist_db.h"

namespace sim {

class mnist_io : public io_entity, public mdb::mnist_db {

    std::function<std::span<float>()> output_mem {};
    std::size_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    std::mt19937 rgen;
    std::uniform_real_distribution<float> dis;

public:
    explicit mnist_io(
        std::size_t _id,
        int _change_interval,
        const std::string &img_file,
        const std::string &label_file,
        int rnd_seed = 0
    );

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] std::size_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; };

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] int& get_change_interval() { return change_interval; }
};

} // sim

#endif //MNIST_IO_H
