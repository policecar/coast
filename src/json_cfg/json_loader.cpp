//
// JSON config file loader implementation
//

#include <nlohmann/json.hpp>
#include <fstream>

#include "ngm2_json.h"
#include "sim_json.h"
#include "dataset_json.h"
#include "gui_vis_json.h"

#include "json_loader.h"

using json = nlohmann::ordered_json;

namespace loader {

json_loader::json_loader(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    json j = json::parse(file);

    global_cfg = j;

    // Handle optional fields that aren't in the NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE macro
    if (j.contains("snapshot_dir")) {
        global_cfg.snapshot_dir = j.at("snapshot_dir").get<std::string>();
    }
}



}
