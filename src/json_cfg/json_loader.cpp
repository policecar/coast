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

using json = nlohmann::json;

namespace loader {

json_loader::json_loader(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    json j = json::parse(file);

    global_cfg = j;

}



}
