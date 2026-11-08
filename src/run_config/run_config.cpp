#include "run_config/run_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <system_error>

namespace coast::cfg {

namespace {

using cfg_json = nlohmann::ordered_json; // ordered: stable key order on write

cfg_json::json_pointer to_pointer(std::string_view ptr)
{
    try {
        return cfg_json::json_pointer(std::string(ptr));
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid JSON pointer '" + std::string(ptr) +
                                 "': " + e.what());
    }
}

} // namespace

// ------------------------------------------------------------ run_config::impl
struct run_config::impl {
    cfg_json    j;
    std::string run_dir;    // "" until set; simc::run() defaults it
    std::string base_path;  // the file config() loaded — stem seeds the default run dir
    int         seed = 0;
};

run_config::run_config() : p_(std::make_unique<impl>()) {}
run_config::~run_config() = default;
run_config::run_config(run_config&&) noexcept            = default;
run_config& run_config::operator=(run_config&&) noexcept = default;

run_config::run_config(const run_config& o) : p_(std::make_unique<impl>(*o.p_)) {}
run_config& run_config::operator=(const run_config& o)
{
    if (this != &o) p_ = std::make_unique<impl>(*o.p_);
    return *this;
}

run_config& run_config::set(std::string_view ptr, double v)
{
    p_->j[to_pointer(ptr)] = v;
    return *this;
}
run_config& run_config::set(std::string_view ptr, std::int64_t v)
{
    p_->j[to_pointer(ptr)] = v;
    return *this;
}
run_config& run_config::set(std::string_view ptr, bool v)
{
    p_->j[to_pointer(ptr)] = v;
    return *this;
}
run_config& run_config::set(std::string_view ptr, std::string_view v)
{
    p_->j[to_pointer(ptr)] = std::string(v);
    return *this;
}

double run_config::get_number(std::string_view ptr) const
{
    try {
        return p_->j.at(to_pointer(ptr)).get<double>();
    } catch (const std::exception& e) {
        throw std::runtime_error("get_number('" + std::string(ptr) +
                                 "'): " + e.what());
    }
}
std::string run_config::get_string(std::string_view ptr) const
{
    try {
        return p_->j.at(to_pointer(ptr)).get<std::string>();
    } catch (const std::exception& e) {
        throw std::runtime_error("get_string('" + std::string(ptr) +
                                 "'): " + e.what());
    }
}

run_config& run_config::run_dir(std::string_view d)
{
    p_->run_dir = std::string(d);
    // Wire outputs into the run directory now, so they show up in dump(). NB:
    // simc::run() re-derives observe/output_dir and snapshot_dir from the resolved
    // run dir (as ABSOLUTE paths) when it writes config.json, so a manual set() of
    // those two pointers after this call is visible in dump() but does NOT survive
    // into the written config — run() has the final say on them.
    p_->j[to_pointer("/observe/output_dir")] = p_->run_dir + "/observe";
    p_->j[to_pointer("/snapshot_dir")]       = p_->run_dir + "/snapshots";
    return *this;
}

run_config& run_config::seed(int v)
{
    p_->seed = v;
    return *this;
}

std::string run_config::dump() const { return p_->j.dump(2); }

const std::string& run_config::base_path()    const { return p_->base_path; }
const std::string& run_config::run_dir_hint() const { return p_->run_dir; }
int                run_config::seed_value()   const { return p_->seed; }

// -------------------------------------------------------------------- config()
run_config config(const std::filesystem::path& base)
{
    std::error_code ec;
    if (!std::filesystem::exists(base, ec)) {
        throw std::runtime_error("config: base file not found: " + base.string());
    }
    std::ifstream in(base);
    if (!in) {
        throw std::runtime_error("config: cannot open: " + base.string());
    }
    run_config c;
    try {
        c.p_->j = cfg_json::parse(in);
    } catch (const std::exception& e) {
        throw std::runtime_error("config: invalid JSON in " + base.string() +
                                 ": " + e.what());
    }
    c.p_->base_path = base.string();
    return c;
}

} // namespace coast::cfg
