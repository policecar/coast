//
// Created for the COAST probe system
//

#include "core/entity_element.h"
#include "sim_core/io_entity.h"

namespace core {

void entity_element::probe(std::string_view name, double value) {
    if (!root_entity_) return;
    root_entity_->probes().record(this, name, value);
}

} // namespace core
