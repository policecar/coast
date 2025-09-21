//
// Created by jk on 03.09.25.
//

#ifndef SIM_IO_ENTITY_H
#define SIM_IO_ENTITY_H

#include <functional>
#include <span>
#include <string>
#include "io_buffer.h"

namespace sim {

struct io_entity {

    virtual ~io_entity() = default;

    virtual void set_outp_func(std::function<std::span<float>()>) {}
    virtual void set_inp_func(std::size_t, const std::function<io_buffer::inp_buf_t()>&) {}

    virtual void process() = 0;

    [[nodiscard]] virtual std::size_t get_outp_id() const = 0;
    [[nodiscard]] virtual std::size_t get_outp_size() const = 0;
    [[nodiscard]] virtual std::span<const std::size_t> get_inp_ids() const = 0;

    [[nodiscard]] virtual std::string status_str() const { return ""; }
};


}

#endif //SIM_IO_ENTITY_H
