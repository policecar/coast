#include "h5browse/h5b_error.h"

#include <hdf5.h>

namespace h5b {

namespace {

struct walk_ctx {
    std::string deepest;
};

// H5E_WALK_UPWARD visits the deepest (most specific) frame first.
extern "C" herr_t deepest_frame_cb(unsigned n, const H5E_error2_t* err,
                                   void* client)
{
    if (n != 0 || err == nullptr) return 0;
    auto* ctx = static_cast<walk_ctx*>(client);
    if (err->desc != nullptr) ctx->deepest = err->desc;
    // Only frame 0 is kept. A non-negative return continues the walk (a
    // negative one would abort it); the n!=0 guard above no-ops every
    // later frame, so the walk is bounded by the guard, not this return.
    return 0;
}

} // namespace

h5_error_silencer::h5_error_silencer()
{
    H5E_auto2_t func = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &func, &data_);
    func_ = reinterpret_cast<void*>(func);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
}

h5_error_silencer::~h5_error_silencer()
{
    H5Eset_auto2(H5E_DEFAULT, reinterpret_cast<H5E_auto2_t>(func_), data_);
}

std::string harvest_h5_error(std::string_view fallback)
{
    walk_ctx ctx;
    H5Ewalk2(H5E_DEFAULT, H5E_WALK_UPWARD, deepest_frame_cb, &ctx);
    if (ctx.deepest.empty()) return std::string{fallback};

    // Recognized failure classes from the T3 corpus -> triage prefixes.
    std::string_view d = ctx.deepest;
    std::string      prefix;
    if (d.find("unable to lock") != std::string_view::npos ||
        d.find("file is already open") != std::string_view::npos) {
        prefix = "file is currently being written (HDF5 file lock held) — ";
    } else if (d.find("truncated file") != std::string_view::npos) {
        prefix = "file is truncated (crashed run or incomplete copy) — ";
    } else if (d.find("signature not found") != std::string_view::npos) {
        prefix = "not an HDF5 file — ";
    }
    return prefix + ctx.deepest;
}

} // namespace h5b
