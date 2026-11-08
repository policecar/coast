//
// hdf5::handle — move-only RAII over hid_t, with off-thread destruction.
//
// A handle holds an HDF5 identifier and a non-owning pointer to the
// executor that owns the worker thread allowed to touch libhdf5. The
// destructor posts the appropriate H5*close call as an async job to that
// executor — so a handle physically cannot be closed on the wrong thread.
//
// Construct it inside an executor job (submit / submit_sync), from the id that
// job's H5*create / H5*open returned.
//
// Move-only by design. Shared ownership of an HDF5 handle would multiply
// the surface for thread-of-destruction mistakes without any concrete
// need; if you think you need it, redesign the data flow instead.
//
#ifndef COAST_HDF5_HANDLE_H
#define COAST_HDF5_HANDLE_H

#include "hdf5/executor.h"

#include <hdf5.h>

namespace hdf5 {

template <auto CloseFn>
class handle {
public:
    handle() = default;

    handle(hid_t id, executor& exec) noexcept
        : id_(id), exec_(&exec) {}

    handle(const handle&)            = delete;
    handle& operator=(const handle&) = delete;

    handle(handle&& o) noexcept
        : id_(o.id_), exec_(o.exec_)
    {
        o.id_   = H5I_INVALID_HID;
        o.exec_ = nullptr;
    }

    handle& operator=(handle&& o) noexcept
    {
        if (this != &o) {
            close_();
            id_     = o.id_;
            exec_   = o.exec_;
            o.id_   = H5I_INVALID_HID;
            o.exec_ = nullptr;
        }
        return *this;
    }

    ~handle() { close_(); }

    [[nodiscard]] hid_t id() const noexcept { return id_; }
    [[nodiscard]] bool  valid() const noexcept { return id_ >= 0 && exec_ != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

    // Hand the executor back so callers can submit further ops against
    // this handle (reads, writes, extensions). Asserts via nullptr only
    // when the handle is in the default-constructed / moved-from state,
    // which is a usage error.
    [[nodiscard]] executor& exec() const noexcept { return *exec_; }

    // Release ownership without closing. Useful when libhdf5 itself takes
    // ownership (rare; e.g. when wrapping a built-in type id we promised
    // not to close). The caller becomes responsible.
    [[nodiscard]] hid_t release() noexcept
    {
        hid_t id = id_;
        id_      = H5I_INVALID_HID;
        exec_    = nullptr;
        return id;
    }

    // Close now (posts the close job) and become empty. Equivalent to
    // moving an empty handle into this one, but reads more clearly at
    // the call site when shutdown order matters.
    void reset(bool immediately = false) noexcept { close_(immediately); }

private:
    void close_(bool immediately = false) noexcept
    {
        if (id_ >= 0 && exec_ != nullptr) {
            const hid_t id = id_;
            // Fire-and-forget close. The executor preserves submission
            // order, so children declared after their parents (the
            // natural RAII order) close before them.
            if (immediately) {
                CloseFn(id);
            } else {
                exec_->submit([id] { CloseFn(id); });
            }
        }
        id_   = H5I_INVALID_HID;
        exec_ = nullptr;
    }

    hid_t     id_   = H5I_INVALID_HID;
    executor* exec_ = nullptr;
};

// Concrete handle types, one per HDF5 close function. The `_h` suffix
// distinguishes the move-only ID-owning handle from the higher-level
// stateful wrappers in this same namespace (e.g. `hdf5::file` for
// extensible-dataset bookkeeping vs. `hdf5::file_h` for the raw
// hid_t-with-RAII-close).
using file_h    = handle<H5Fclose>;
using group_h   = handle<H5Gclose>;
using dataset_h = handle<H5Dclose>;
using space_h   = handle<H5Sclose>;
using plist_h   = handle<H5Pclose>;
using attr_h    = handle<H5Aclose>;
using type_h    = handle<H5Tclose>;

} // namespace hdf5

#endif // COAST_HDF5_HANDLE_H
