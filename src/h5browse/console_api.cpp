#include "h5browse/console_api.h"

#include "h5browse/chunk_cache.h"
#include "h5browse/dataset_reader.h"
#include "h5browse/file_model.h"
#include "hdf5/executor.h"

#include <atomic>
#include <cctype>
#include <cstring>
#include <utility>

namespace h5b::console {

namespace {

// Widen one native-dtype element at `p` to double (the console's lingua
// franca). Reuses the scalar_type.h type dispatch.
double to_double(const std::byte* p, scalar_type t)
{
    return dispatch(t, [p](auto id) -> double {
        using elem_t = typename decltype(id)::type;
        elem_t v;
        std::memcpy(&v, p, sizeof(elem_t));
        return static_cast<double>(v);
    });
}

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

} // namespace

// --------------------------------------------------------------- range_state

// Heap-held iteration state behind row_range::iterator. Owns the read_pipeline;
// `desc` is stored by value because the pipeline keeps a reference to it (and
// must be declared before `pipe` so it outlives it).
struct range_state {
    const file_console*            fc = nullptr;
    dataset_desc                   desc;
    std::uint32_t                  dataset_index = 0;
    std::unique_ptr<read_pipeline> pipe;

    std::span<const std::byte> data;        // current batch bytes (until next fetch)
    std::uint64_t              batch_first = 0;
    std::uint64_t              batch_rows  = 0;
    std::uint64_t              cursor      = 0; // row within the current batch
    bool                       done        = false;

    void fetch_next_batch()
    {
        auto b = pipe->next();
        if (!b) {
            done = true;
            fail("scan read failed: " + b.error().message);
        }
        if (!b->has_value()) { // drained
            done       = true;
            data       = {};
            batch_rows = 0;
            return;
        }
        const auto& batch = **b;
        data        = batch.data;
        batch_first = batch.first_row;
        batch_rows  = batch.n_rows;
        cursor      = 0;
    }

    [[nodiscard]] row_view current() const
    {
        const std::size_t esz = scalar_size(desc.dtype);
        return row_view{data.data() + cursor * desc.row_width * esz,
                        batch_first + cursor, desc.row_width, desc.dtype,
                        &desc.shape, nullptr};
    }
};

// ------------------------------------------------------------ row_range::iterator

row_view row_range::iterator::operator*() const
{
    if (!st_ || st_->done) fail("dereferencing a past-the-end row iterator");
    return st_->current();
}

row_range::iterator& row_range::iterator::operator++()
{
    if (!st_ || st_->done) return *this;
    ++st_->cursor;
    if (st_->cursor >= st_->batch_rows) st_->fetch_next_batch();
    return *this;
}

bool row_range::iterator::operator==(std::default_sentinel_t) const
{
    return !st_ || st_->done;
}

// -------------------------------------------------------------------- row_range

row_range::iterator row_range::begin() const
{
    auto st           = std::make_shared<range_state>();
    st->fc            = fc_;
    st->desc          = fc_->desc(dataset_);
    st->dataset_index = dataset_;
    st->pipe          = std::make_unique<read_pipeline>(*fc_->exec_, fc_->file_,
                                                        st->desc, first_, n_);
    st->fetch_next_batch(); // primes the first batch (or marks done if empty)

    iterator it;
    it.st_ = std::move(st);
    return it;
}

// --------------------------------------------------------------- row_view::at

double row_view::at(std::uint64_t col) const
{
    if (col >= width_) fail("row_view::at(): column index out of range");
    return to_double(data_ + col * scalar_size(dtype_), dtype_);
}

// ---------------------------------------------------------------- dataset_handle

std::string_view dataset_handle::name() const
{
    return fc_->desc(dataset_).display_name;
}
std::uint64_t dataset_handle::rows() const { return fc_->desc(dataset_).rows; }
std::uint64_t dataset_handle::width() const
{
    return fc_->desc(dataset_).row_width;
}
scalar_type dataset_handle::dtype() const { return fc_->desc(dataset_).dtype; }

row_range dataset_handle::scan() const
{
    return scan(0, fc_->desc(dataset_).rows);
}

row_range dataset_handle::scan(std::uint64_t first, std::uint64_t n) const
{
    row_range r;
    r.fc_      = fc_;
    r.dataset_ = dataset_;
    r.first_   = first;
    r.n_       = n;
    return r;
}

row_view dataset_handle::row(std::uint64_t i) const
{
    const dataset_desc& d = fc_->desc(dataset_);
    if (i >= d.rows) fail("dataset_handle::row(): row index out of range");

    const std::uint64_t rpb     = default_block_rows(d);
    const std::uint64_t blk_idx = i / rpb;

    auto blk = fc_->cache_->get(dataset_, blk_idx);
    if (!blk) {
        auto rc = read_block(*fc_->exec_, fc_->file_, d, dataset_, blk_idx, rpb);
        if (!rc) fail("dataset_handle::row(): " + rc.error().message);
        blk = *rc;
        fc_->cache_->put(blk);
    }

    const std::uint64_t off = i - blk->first_row;
    const std::size_t   esz = scalar_size(d.dtype);
    return row_view{blk->bytes.data() + off * d.row_width * esz, i, d.row_width,
                    d.dtype, &d.shape, blk};
}

std::vector<double> dataset_handle::column(std::uint64_t col) const
{
    const dataset_desc& d = fc_->desc(dataset_);
    if (col >= d.row_width) fail("dataset_handle::column(): column out of range");

    std::vector<double> out;
    out.reserve(d.rows);
    const std::size_t esz = scalar_size(d.dtype);

    read_pipeline pipe(*fc_->exec_, fc_->file_, d, 0, d.rows);
    for (;;) {
        auto b = pipe.next();
        if (!b) fail("dataset_handle::column(): " + b.error().message);
        if (!b->has_value()) break;
        const auto&      batch = **b;
        const std::byte* base  = batch.data.data();
        for (std::uint64_t r = 0; r < batch.n_rows; ++r) {
            out.push_back(to_double(base + (r * d.row_width + col) * esz, d.dtype));
        }
    }
    return out;
}

std::vector<std::uint64_t> dataset_handle::steps() const
{
    const dataset_desc& d = fc_->desc(dataset_);
    if (!d.step_index_path) return {};

    // Synthetic desc for the sibling step-index dataset (u64, scalar rows).
    // HDF5 converts the on-disk type to native u64 on read, so this is safe
    // even if the writer stored a different integer width.
    dataset_desc sd;
    sd.h5_path      = *d.step_index_path;
    sd.display_name = "step_index";
    sd.dtype        = scalar_type::u64;
    sd.rows         = d.rows;
    sd.row_width    = 1;

    std::vector<std::uint64_t> out;
    out.reserve(d.rows);

    read_pipeline pipe(*fc_->exec_, fc_->file_, sd, 0, d.rows);
    for (;;) {
        auto b = pipe.next();
        if (!b) fail("dataset_handle::steps(): " + b.error().message);
        if (!b->has_value()) break;
        const auto* p = reinterpret_cast<const std::uint64_t*>((*b)->data.data());
        out.insert(out.end(), p, p + (*b)->n_rows);
    }
    return out;
}

// ------------------------------------------------------------------ file_console

file_console::file_console(hdf5::executor&              exec,
                           const std::filesystem::path& path)
    : exec_(&exec), cache_(std::make_unique<chunk_cache>())
{
    auto m = open_and_map(exec, path);
    if (!m) {
        fail("cannot open '" + path.string() + "': " + m.error().message);
    }
    model_ = std::make_shared<const file_model>(std::move(*m));

    auto fh = open_file_readonly(exec, path);
    if (!fh) {
        fail("cannot open read handle for '" + path.string() +
             "': " + fh.error().message);
    }
    file_ = *fh;
}

file_console::~file_console()
{
    if (file_ >= 0 && exec_ != nullptr) close_file_handle(*exec_, file_);
}

const dataset_desc& file_console::desc(std::uint32_t index) const
{
    if (index >= model_->datasets.size()) {
        fail("dataset index out of range");
    }
    return model_->datasets[index];
}

std::vector<std::string> file_console::list() const
{
    std::vector<std::string> out;
    out.reserve(model_->datasets.size());
    for (const auto& d : model_->datasets) out.push_back(d.display_name);
    return out;
}

std::size_t file_console::count() const { return model_->datasets.size(); }

std::string_view file_console::family() const
{
    return file_family_name(model_->family);
}

const std::filesystem::path& file_console::path() const { return model_->path; }

dataset_handle file_console::dataset(std::string_view name) const
{
    for (std::uint32_t i = 0; i < model_->datasets.size(); ++i) {
        if (model_->datasets[i].display_name == name) {
            dataset_handle h;
            h.fc_      = this;
            h.dataset_ = i;
            return h;
        }
    }
    fail("no dataset named '" + std::string(name) + "'");
}

dataset_handle file_console::dataset(std::uint32_t index) const
{
    (void)desc(index); // bounds check (throws on OOR)
    dataset_handle h;
    h.fc_      = this;
    h.dataset_ = index;
    return h;
}

// ------------------------------------------------------------------ open() façade

namespace {
// The one executor open() builds file_consoles on. Set at startup by the host
// (set_executor) and cleared when that host tears down (clear_executor). Not
// owning: the executor is a longer-lived object (main()-scope in COAST) that the
// registrant must keep alive for as long as open() may be called.
//
// Atomic so the null check in open() and the registrant's store/clear are free
// of data races even if they ever land on different threads. It still assumes a
// single registrant at a time: set_executor overwrites unconditionally, and
// clear_executor is compare-and-clear so a teardown cannot wipe a DIFFERENT
// executor that a later registrant installed. Once cleared, open() fails cleanly
// with "no executor registered" instead of dereferencing a dangling pointer.
std::atomic<hdf5::executor*> g_exec{nullptr};
} // namespace

void set_executor(hdf5::executor& exec)
{
    g_exec.store(&exec, std::memory_order_release);
}

void clear_executor(hdf5::executor& exec)
{
    // Only clear if we still point at THIS executor — a later set_executor() with
    // a different one must survive this registrant's teardown.
    hdf5::executor* expected = &exec;
    g_exec.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

std::shared_ptr<file_console> open(const std::filesystem::path& path)
{
    hdf5::executor* exec = g_exec.load(std::memory_order_acquire);
    if (exec == nullptr) {
        throw std::runtime_error(
            "h5b::console::open(): no HDF5 executor registered "
            "(set_executor() was never called)");
    }
    // make_shared runs here, in compiled coast_lib: the control block and its
    // vtable live in the host image, so a handle a macro stores survives the
    // interpreter that created the variable (the JIT-lifetime hazard is avoided
    // at the source rather than only papered over by the retaining arena).
    return std::make_shared<file_console>(*exec, path);
}

std::string var_name_for(const std::filesystem::path& path)
{
    const std::string stem = path.stem().string();
    // Always "f_"-prefixed: it keeps the name a valid identifier for any stem
    // (leading digit, empty stem) and reads as "a file handle" at a glance.
    std::string id = "f_";
    id.reserve(stem.size() + 2);
    for (char c : stem) {
        const auto uc = static_cast<unsigned char>(c);
        id.push_back((std::isalnum(uc) != 0) ? c : '_');
    }
    return id;
}

} // namespace h5b::console
