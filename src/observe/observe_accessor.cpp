//
// Created for the COAST observe system
//

#include "observe/observe_accessor.h"
#include "observe/observe_registry.h"

namespace observe {

void observe_accessor::emit(core::entity_element&    e,
                            std::string_view         group,
                            std::string_view         id,
                            hid_t                    h5_dtype,
                            std::span<const hsize_t> shape,
                            const void*              value,
                            std::size_t              value_bytes)
{
    reg_.emit(e, group, id, h5_dtype, shape, value, value_bytes);
}

bool observe_accessor::group_active(std::string_view group) const {
    return reg_.group_active(group);
}

void observe_accessor::trigger(std::string_view group, trigger_mode_t mode, bool cond) {
    if (cond) reg_.trigger(group, mode);
}

} // namespace observe
