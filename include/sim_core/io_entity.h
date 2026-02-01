//
// Created by jk on 03.09.25.
//

#ifndef SIM_IO_ENTITY_H
#define SIM_IO_ENTITY_H

#include <functional>
#include <span>
#include <string>
#include "io_buffer.h"
#include "core/entity_element.h"
#include "core/structure_registry.h"
#include "probe/entity_probe_set.h"

namespace sim {

using output_id_t = std::size_t;
using output_channel_t = std::pair<output_id_t,std::size_t>;

struct io_entity : public core::entity_element {

    io_entity() = default;
    ~io_entity() override = default;

    // Non-copyable (structure_registry is non-copyable)
    io_entity(const io_entity&) = delete;
    io_entity& operator=(const io_entity&) = delete;

    // Movable
    io_entity(io_entity&&) = default;
    io_entity& operator=(io_entity&&) = default;

    virtual void set_outp_func(std::function<std::span<float>()>) {}
    virtual void set_inp_func(std::size_t, const std::function<io_buffer::inp_buf_t()>&) {}

    virtual void process() = 0;

    [[nodiscard]] virtual output_id_t get_outp_id() const = 0;
    [[nodiscard]] virtual std::size_t get_outp_size() const = 0;
    [[nodiscard]] virtual std::span<const std::size_t> get_inp_ids() const = 0;

    [[nodiscard]] virtual std::string status_str() const { return ""; }

    // -- Entity element hierarchy support --

    /// Access the structure registry for this entity.
    [[nodiscard]] core::structure_registry& structure() { return structure_; }
    [[nodiscard]] const core::structure_registry& structure() const { return structure_; }

    /// Access the probe set for this entity.
    [[nodiscard]] probe::entity_probe_set& probes() { return probes_; }
    [[nodiscard]] const probe::entity_probe_set& probes() const { return probes_; }

    /// Initialize the structure registry and probe set.
    /// Call this after all child elements have been created, and after the entity
    /// is in its final memory location (not subject to vector reallocation).
    void init_structure() {
        structure_.build(this);
        probes_.init(this, &structure_);
    }

protected:
    core::structure_registry structure_;
    probe::entity_probe_set probes_;
};


}

#endif //SIM_IO_ENTITY_H
