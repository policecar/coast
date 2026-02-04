//
// State Logger implementation
//

#include "logging/state_logger.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"

#include <chrono>

namespace logging {

// --- hdf5_state_writer implementation ---

hdf5_state_writer::hdf5_state_writer(hdf5::file& file)
    : file_(file)
{
}

hdf5_state_writer::~hdf5_state_writer()
{
    // Close any remaining groups
    group_stack_.clear();
}

hid_t hdf5_state_writer::current_loc() const
{
    if (group_stack_.empty()) {
        return file_.id();
    }
    return group_stack_.back().id();
}

void hdf5_state_writer::begin_group(std::string_view name)
{
    std::string name_str(name);
    auto g = hdf5::group::create(current_loc(), name_str.c_str());
    group_stack_.push_back(std::move(g));
}

void hdf5_state_writer::end_group()
{
    if (!group_stack_.empty()) {
        group_stack_.pop_back();
    }
}

void hdf5_state_writer::write(std::string_view name, double value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_DOUBLE, space.id());
    ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, float value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_FLOAT, space.id());
    ds.write(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, std::int32_t value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_INT32, space.id());
    ds.write(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, std::int64_t value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_INT64, space.id());
    ds.write(H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, std::uint32_t value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_UINT32, space.id());
    ds.write(H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, std::uint64_t value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_UINT64, space.id());
    ds.write(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, &value);
}

void hdf5_state_writer::write(std::string_view name, bool value)
{
    std::string name_str(name);
    auto space = hdf5::dataspace::scalar();
    // HDF5 doesn't have native bool, use int8
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_INT8, space.id());
    std::int8_t v = value ? 1 : 0;
    ds.write(H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, &v);
}

void hdf5_state_writer::write(std::string_view name, std::span<const float> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_FLOAT, space.id());
    ds.write(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write(std::string_view name, std::span<const double> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_DOUBLE, space.id());
    ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write(std::string_view name, std::span<const std::int32_t> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_INT32, space.id());
    ds.write(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write(std::string_view name, std::span<const std::uint32_t> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_UINT32, space.id());
    ds.write(H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write(std::string_view name, std::span<const std::uint16_t> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_UINT16, space.id());
    ds.write(H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write(std::string_view name, std::span<const std::uint8_t> values)
{
    if (values.empty()) return;
    std::string name_str(name);
    hsize_t dims[1] = {values.size()};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_UINT8, space.id());
    ds.write(H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write_2d(std::string_view name, std::span<const float> values,
                                  std::size_t rows, std::size_t cols)
{
    if (values.empty() || rows == 0 || cols == 0) return;
    std::string name_str(name);
    hsize_t dims[2] = {rows, cols};
    auto space = hdf5::dataspace::simple(std::span(dims));
    auto ds = hdf5::dataset::create(current_loc(), name_str.c_str(), H5T_NATIVE_FLOAT, space.id());
    ds.write(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, values.data());
}

void hdf5_state_writer::write_attr(std::string_view name, std::string_view value)
{
    std::string name_str(name);
    hdf5::write_attr(current_loc(), name_str.c_str(), value);
}

// --- hdf5_state_reader implementation ---

hdf5_state_reader::hdf5_state_reader(hdf5::file& file)
    : file_(file)
{
}

hdf5_state_reader::~hdf5_state_reader()
{
    group_stack_.clear();
}

hid_t hdf5_state_reader::current_loc() const
{
    if (group_stack_.empty()) {
        return file_.id();
    }
    return group_stack_.back().id();
}

bool hdf5_state_reader::enter_group(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return false;
    }
    auto g = hdf5::group::open(current_loc(), name_str.c_str());
    if (!g) return false;
    group_stack_.push_back(std::move(g));
    return true;
}

void hdf5_state_reader::leave_group()
{
    if (!group_stack_.empty()) {
        group_stack_.pop_back();
    }
}

bool hdf5_state_reader::has_field(std::string_view name)
{
    std::string name_str(name);
    return hdf5::link_exists(current_loc(), name_str.c_str());
}

double hdf5_state_reader::read_double(std::string_view name, double default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    double value;
    ds.read(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, &value);
    return value;
}

float hdf5_state_reader::read_float(std::string_view name, float default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    float value;
    ds.read(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, &value);
    return value;
}

std::int32_t hdf5_state_reader::read_int32(std::string_view name, std::int32_t default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    std::int32_t value;
    ds.read(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, &value);
    return value;
}

std::int64_t hdf5_state_reader::read_int64(std::string_view name, std::int64_t default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    std::int64_t value;
    ds.read(H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, &value);
    return value;
}

std::uint32_t hdf5_state_reader::read_uint32(std::string_view name, std::uint32_t default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    std::uint32_t value;
    ds.read(H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, &value);
    return value;
}

std::uint64_t hdf5_state_reader::read_uint64(std::string_view name, std::uint64_t default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    std::uint64_t value;
    ds.read(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, &value);
    return value;
}

bool hdf5_state_reader::read_bool(std::string_view name, bool default_value)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return default_value;
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return default_value;
    std::int8_t value;
    ds.read(H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, &value);
    return value != 0;
}

std::vector<float> hdf5_state_reader::read_float_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<float> values(dims[0]);
    ds.read(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<double> hdf5_state_reader::read_double_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<double> values(dims[0]);
    ds.read(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<std::int32_t> hdf5_state_reader::read_int32_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<std::int32_t> values(dims[0]);
    ds.read(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<std::uint32_t> hdf5_state_reader::read_uint32_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<std::uint32_t> values(dims[0]);
    ds.read(H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<std::uint16_t> hdf5_state_reader::read_uint16_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<std::uint16_t> values(dims[0]);
    ds.read(H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<std::uint8_t> hdf5_state_reader::read_uint8_vec(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.empty()) return {};

    std::vector<std::uint8_t> values(dims[0]);
    ds.read(H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::vector<float> hdf5_state_reader::read_float_2d(std::string_view name, array_2d_info& info)
{
    std::string name_str(name);
    info = {};

    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return {};
    }
    auto ds = hdf5::dataset::open(current_loc(), name_str.c_str());
    if (!ds) return {};

    auto space = hdf5::dataspace::from_dataset(ds.id());
    auto dims = space.dims();
    if (dims.size() < 2) return {};

    info.rows = dims[0];
    info.cols = dims[1];

    std::vector<float> values(info.rows * info.cols);
    ds.read(H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, values.data());
    return values;
}

std::string hdf5_state_reader::read_attr(std::string_view name)
{
    std::string name_str(name);
    // Check if attribute exists first to avoid HDF5 error output
    if (!hdf5::attr_exists(current_loc(), name_str.c_str())) {
        return "";
    }
    auto result = hdf5::read_attr_string(current_loc(), name_str.c_str());
    return result.value_or("");
}

// --- state_logger implementation ---

log_result state_logger::save_state(sim::env& env, std::uint64_t step_count,
                                     const state_save_config& config)
{
    auto file = hdf5::file::create(config.output_path.string());
    if (!file) {
        return log_result::error("Failed to create state file: " + config.output_path.string());
    }

    hdf5_state_writer writer(file);

    // Write metadata
    writer.begin_group("metadata");
    writer.write_attr("schema_version", SCHEMA_VERSION);
    writer.write_attr("type", std::string_view("simulation_state"));
    writer.write("step_count", step_count);

    if (!config.config_json.empty()) {
        writer.write_attr("config", std::string_view(config.config_json));
    }

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time_t);
    if (!timestamp.empty() && timestamp.back() == '\n') {
        timestamp.pop_back();
    }
    writer.write_attr("timestamp", std::string_view(timestamp));
    writer.end_group();

    // Save IO buffer states
    writer.begin_group("io_buffers");
    auto& buffers = env.get_io_buffers();
    for (auto& [id, buffer] : buffers) {
        writer.begin_group(std::to_string(id));
        auto data = buffer.cur_read_buffer();
        writer.write("data", std::span<const float>(data.data(), data.size()));
        writer.end_group();
    }
    writer.end_group();

    // Save entity states
    writer.begin_group("entities");
    for (auto& entity : env.iterate_entities()) {
        writer.begin_group(std::to_string(entity.get_outp_id()));
        writer.write_attr("type", entity.type_id().name());
        entity.save_state(writer);
        writer.end_group();
    }
    writer.end_group();

    file.flush();
    return log_result::ok();
}

state_load_result state_logger::load_state(sim::env& env, const std::filesystem::path& path)
{
    state_load_result result;

    auto file = hdf5::file::open_read(path.string());
    if (!file) {
        result.error_message = "Failed to open state file: " + path.string();
        return result;
    }

    hdf5_state_reader reader(file);

    // Read metadata
    if (!reader.enter_group("metadata")) {
        result.error_message = "Missing metadata group in state file";
        return result;
    }

    std::string version = reader.read_attr("schema_version");
    if (version != SCHEMA_VERSION) {
        result.warnings.push_back("Schema version mismatch: file=" + version +
                                   ", expected=" + std::string(SCHEMA_VERSION));
    }

    result.restored_step = reader.read_uint64("step_count", 0);
    result.config_json = reader.read_attr("config");
    reader.leave_group();

    // Load IO buffer states
    if (reader.enter_group("io_buffers")) {
        auto& buffers = env.get_io_buffers();
        for (auto& [id, buffer] : buffers) {
            std::string id_str = std::to_string(id);
            if (reader.enter_group(id_str)) {
                auto data = reader.read_float_vec("data");
                auto write_buf = buffer.cur_write_buffer();
                if (data.size() == write_buf.size()) {
                    std::copy(data.begin(), data.end(), write_buf.begin());
                } else {
                    result.warnings.push_back("Buffer " + id_str + " size mismatch");
                }
                reader.leave_group();
            }
        }
        reader.leave_group();
    }

    // Load entity states
    if (reader.enter_group("entities")) {
        for (auto& entity : env.iterate_entities()) {
            std::string id_str = std::to_string(entity.get_outp_id());
            if (reader.enter_group(id_str)) {
                entity.load_state(reader);
                reader.leave_group();
            }
        }
        reader.leave_group();
    }

    result.success = true;
    return result;
}

} // namespace logging
