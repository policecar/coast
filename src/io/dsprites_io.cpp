//
// dSprites dataset IO entity implementation
//

#include <format>
#include <hdf5.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <ranges>
#include <span>

#include "io/dsprites_io.h"
#include "diag/diag.h"
#include "core/require.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <chrono>

namespace dsprites {

namespace {

// RAII wrappers for HDF5 handles. Each destructor calls H5*close
// directly, so the handles must be constructed AND destroyed on the
// executor's worker thread — i.e. their entire lifetime sits inside a
// single submit_sync lambda. See dsprites_io ctor.
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

dsprites_io::dsprites_io(const dsprites_param_t& params, sim::output_id_t output_id,
                         hdf5::executor& h5_exec) :
    params(params),
    id(output_id),
    // image_count is only known after the HDF5 read below, so the sequencer is
    // (re)built with the real count once image_count is set.
    seq_(0, static_cast<int>(params.change_interval), params.shuffle,
         std::to_underlying(params.random_seed)),
    rgen(std::to_underlying(params.random_seed)),
    dis(params.noise_range.first, params.noise_range.second)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
    // Surface a bad manifest path before the ~900 MB HDF5 read; the manifest
    // is parsed against the real image_count below.
    dataset::ensure_manifest_readable(params.index_manifest);

    // Bulk read of the dSprites image set. All libhdf5 work — open,
    // dataspace inspection, the full H5Dread, and the implicit closes
    // of the local RAII wrappers — runs on the executor worker thread
    // inside one submit_sync, so the surrounding setup never enters
    // libhdf5 from this thread. The raw byte buffer is moved out of
    // the lambda so the float conversion below stays on the caller.
    std::vector<std::uint8_t> tmp;
    auto t1 = std::chrono::system_clock::now();
    const std::string& hdf5_path = params.hdf5_file;
    h5_exec.submit_sync([this, &tmp, &hdf5_path] {
        hdf5_file file(hdf5_path);
        if (!file) {
            throw std::runtime_error("Cannot open HDF5 file: " + hdf5_path);
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

        // The file declares dims[0]: count it without overflow, refuse a size
        // the file does not store (#159), and read through a memory dataspace
        // sized to the buffer.
        const std::size_t n = hdf5::element_count(dataset.id, dims);
        hdf5::require_stored(dataset.id, n);
        tmp.resize(n);
        hdf5::read_all(dataset.id, H5T_NATIVE_UINT8, tmp.data(), n);
    });
    auto t2 = std::chrono::system_clock::now();

    // Now that image_count is known, build the sequencer for the real set size.
    seq_ = dataset::build_sequencer(params, image_count);

    // convert to float
    images.resize(tmp.size());
    for (auto [fval,ival] : std::views::zip(images,tmp)) {
        fval = static_cast<float>(ival) / 1.0f; // normalizing by 1.0f instead of 255.0f as images are binary
    }
    auto t3 = std::chrono::system_clock::now();

    using namespace std::literals;

    COAST_DEBUG("io", "dSprites load times: decode={}ms convert={}ms",
                (t2 - t1) / 1ms, (t3 - t2) / 1ms);

}

void dsprites_io::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

sim::step_result dsprites_io::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);
    auto outp = output_buf->cur_write_buffer();
    COAST_REQUIRE(outp.size() == image_pixels, core::error_kind::contract_violation);

    if (seq_.change_interval() > 0) {
        // Convert uint8 to float with epsilon normalization
        const std::size_t cur_idx = seq_.current_index();
        COAST_REQUIRE((cur_idx + 1) * image_pixels <= images.size(),
                      core::error_kind::contract_violation);
        auto img = std::span ( images.begin() + cur_idx * image_pixels, image_pixels );
        std::ranges::copy(img,outp.begin());
    } else {
        std::ranges::fill(outp, 0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen), 0.0f, 1.0f);
    }

    // Record which sample is presented this step. This does NOT line up with
    // population/out at equal step_index: IO buffers are double-buffered, so a
    // consumer reads this value one step later -- one step per buffered hop on
    // the feedforward path, and recurrent (attractor) groups have no fixed
    // offset at all. cur_idx is std::size_t (no observe dtype specialisation) -> cast.
    observed_index_ = static_cast<std::uint64_t>(seq_.current_index());  // snapshot before seq_ advances; read in perform_observations

    // Advance to next image
    seq_.advance();

    return {};
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
    status += " | epoch: " + std::to_string(seq_.current_epoch());
    status += " | idx: " + std::to_string(seq_.current_index());
    status += " / " + std::to_string(image_count);
    return status;
}

void dsprites_io::save_state(logging::state_writer& writer) const
{
    dataset::save_sequenced_state(params, writer, seq_, rgen);
}

void dsprites_io::load_state(logging::state_reader& reader)
{
    (void)dataset::restore_sequenced_state(params, reader, seq_, rgen,
                                           std::format("dsprites {}", get_outp_id()));
}

void dsprites_io::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("input"))
        acc.observe(*this, "input", "index", observed_index_);
}

} // namespace sim
