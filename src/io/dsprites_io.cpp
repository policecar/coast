//
// dSprites dataset IO entity implementation
//

#include <hdf5.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <ranges>
#include <span>

#include "dsprites_io.h"
#include "logging/state_io.h"

#include <print>

// temporary
#include <iostream>
#include <chrono>

namespace dsprites {

namespace {

// RAII wrappers for HDF5 handles
struct hdf5_file {
    hid_t id;
    explicit hdf5_file(const std::string& path) : id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) {}
    ~hdf5_file() { if (id >= 0) H5Fclose(id); }
    hdf5_file(const hdf5_file&) = delete;
    hdf5_file& operator=(const hdf5_file&) = delete;
    explicit operator bool() const { return id >= 0; }
};

struct hdf5_dataset {
    hid_t id;
    hdf5_dataset(hid_t file, const char* name) : id(H5Dopen2(file, name, H5P_DEFAULT)) {}
    ~hdf5_dataset() { if (id >= 0) H5Dclose(id); }
    hdf5_dataset(const hdf5_dataset&) = delete;
    hdf5_dataset& operator=(const hdf5_dataset&) = delete;
    explicit operator bool() const { return id >= 0; }
};

struct hdf5_dataspace {
    hid_t id;
    explicit hdf5_dataspace(hid_t dataset) : id(H5Dget_space(dataset)) {}
    ~hdf5_dataspace() { if (id >= 0) H5Sclose(id); }
    hdf5_dataspace(const hdf5_dataspace&) = delete;
    hdf5_dataspace& operator=(const hdf5_dataspace&) = delete;
    explicit operator bool() const { return id >= 0; }
};

} // namespace

dsprites_io::dsprites_io(const dsprites_param_t& params, sim::output_id_t output_id) :
    params(params),
    id(output_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(params.change_interval),
    rgen(params.random_seed),
    dis(params.noise_range.first, params.noise_range.second)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }

    hdf5_file file(params.hdf5_file);
    if (!file) {
        throw std::runtime_error("Cannot open HDF5 file: " + params.hdf5_file);
    }

    hdf5_dataset dataset(file.id, "imgs");
    if (!dataset) {
        throw std::runtime_error("Cannot open dataset 'imgs' in HDF5 file");
    }

    hdf5_dataspace dataspace(dataset.id);
    if (!dataspace) {
        throw std::runtime_error("Failed to get dataspace for 'imgs' dataset");
    }

    int ndims = H5Sget_simple_extent_ndims(dataspace.id);
    if (ndims != 3) {
        throw std::runtime_error("Unexpected dSprites image dimensions");
    }

    hsize_t dims[3];
    if (H5Sget_simple_extent_dims(dataspace.id, dims, nullptr) < 0) {
        throw std::runtime_error("Failed to get dimensions from HDF5 dataspace");
    }
    if (dims[1] != image_dim || dims[2] != image_dim) {
        throw std::runtime_error("Unexpected dSprites image shape");
    }

    image_count = dims[0];
    if (image_count == 0) {
        throw std::runtime_error("dSprites dataset is empty");
    }

    idx_dis = std::uniform_int_distribution<std::size_t>(0,image_count-1);

    auto t1 = std::chrono::system_clock::now();
    std::vector<std::uint8_t> tmp;
    tmp.resize(image_count * image_pixels);
    if (H5Dread(dataset.id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data()) < 0) {
        throw std::runtime_error("Failed to read dataset from HDF5 file");
    }
    auto t2 = std::chrono::system_clock::now();

    // convert to float
    images.resize(tmp.size());
    for (auto [fval,ival] : std::views::zip(images,tmp)) {
        fval = static_cast<float>(ival) / 1.0f; // normalizing by 1.0f instead of 255.0f as images are binary
    }
    auto t3 = std::chrono::system_clock::now();

    using namespace std::literals;

    std::cout << "Times to load: \n\tdecode: " << (t2 - t1) / 1ms << "ms\n\tconvert: " << (t3-t2) / 1ms << "ms\n";

}

void dsprites_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = std::move(function);
}

void dsprites_io::process()
{
    if (!output_mem) {
        throw std::runtime_error("dsprites_io::process() called before set_outp_func()");
    }
    auto outp = output_mem();
    if (outp.size() != image_pixels) {
        throw std::runtime_error("Output span size mismatch: expected " +
            std::to_string(image_pixels) + ", got " + std::to_string(outp.size()));
    }

    if (change_interval > 0) {
        // Convert uint8 to float with epsilon normalization
        auto img = std::span ( images.begin() + cur_idx * image_pixels, image_pixels );
        std::ranges::copy(img,outp.begin());
    } else {
        std::ranges::fill(outp, 0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen), 0.0f, 1.0f);
    }

    // Advance to next image
    if (change_interval > 0 && ++ci_cnt >= static_cast<std::size_t>(change_interval)) {
        ci_cnt = 0;
        if (params.randomize_readout) {
            cur_idx = idx_dis(rgen);
        } else if (++cur_idx >= image_count) {
            cur_idx = 0;
            ++cur_epoch;
        }
    }
}

sim::output_id_t dsprites_io::get_outp_id() const
{
    return id;
}

std::size_t dsprites_io::get_outp_size() const
{
    return image_pixels;
}

std::string dsprites_io::status_str() const
{
    std::string status { "dSprites IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    status += " / " + std::to_string(image_count);
    return status;
}

void dsprites_io::save_state(logging::state_writer& writer) const
{
    writer.write("cur_epoch", static_cast<std::uint64_t>(cur_epoch));
    writer.write("cur_idx", static_cast<std::uint64_t>(cur_idx));
    writer.write("ci_cnt", static_cast<std::uint64_t>(ci_cnt));
}

void dsprites_io::load_state(logging::state_reader& reader)
{
    cur_epoch = static_cast<std::size_t>(reader.read_uint64("cur_epoch", cur_epoch));
    cur_idx = static_cast<std::size_t>(reader.read_uint64("cur_idx", cur_idx));
    ci_cnt = static_cast<std::size_t>(reader.read_uint64("ci_cnt", ci_cnt));
}

} // namespace sim
