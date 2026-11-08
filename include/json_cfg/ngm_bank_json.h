//
// JSON adapters for the shared ngm::bank_t / ngm::input_route_t (ngm/bank.h),
// used by the ngm3 and the ngm2.5 config schema. They must live in namespace
// ngm, the types' own, for nlohmann's ADL lookup to find them.
//

#ifndef COAST_NGM_BANK_JSON_H
#define COAST_NGM_BANK_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"
#include "ngm/bank.h"

namespace ngm {

// COAST_JSON_STRICT_ENUM (json_cfg/strict_enum.h): an unknown string such as
// "proxymal" throws at load instead of silently becoming the first value.
COAST_JSON_STRICT_ENUM( bank_t, {
    {bank_t::apical,   "apical"  },
    {bank_t::proximal, "proximal"},
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(input_route_t,
    channel_name,
    bank
)

} // namespace ngm

#endif // COAST_NGM_BANK_JSON_H
