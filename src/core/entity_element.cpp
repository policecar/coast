//
// Created for the COAST probe system
//

#include "core/entity_element.h"

namespace core {

void entity_element::probe([[maybe_unused]] std::string_view name,
                           [[maybe_unused]] double value) {
    // Implementation deferred to Phase 2 when entity_probe_set is available.
    // Will call: root_entity_->probes().record(this, name, value);
}

} // namespace core
