// open_and_map — runs entirely in one hdf5::executor job (IDD §6).
// Every libhdf5 failure is caught inside the job body and returned as an
// h5b::error value; nothing throws across the executor queue. Every COAST
// log/state file carries a /metadata group stamped with a `type` attribute;
// detection reads /metadata@type uniformly (observe_log / buffer_log),
// rejects simulation_state checkpoints, and reports unknown otherwise.

#include "h5browse/file_model.h"

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <hdf5.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <format>
#include <numeric>
#include <system_error>

namespace h5b {

namespace {

// ---------------------------------------------------------------- helpers
// All of these run on the executor worker thread.

std::optional<scalar_type> classify(hid_t type)
{
    const H5T_class_t cls = H5Tget_class(type);
    const std::size_t sz  = H5Tget_size(type);
    if (cls == H5T_FLOAT) {
        if (sz == 4) return scalar_type::f32;
        if (sz == 8) return scalar_type::f64;
        return std::nullopt;
    }
    if (cls == H5T_INTEGER) {
        const bool sgn = H5Tget_sign(type) == H5T_SGN_2;
        switch (sz) {
            case 1: return sgn ? scalar_type::i8 : scalar_type::u8;
            case 2: return sgn ? scalar_type::i16 : scalar_type::u16;
            case 4: return sgn ? scalar_type::i32 : scalar_type::u32;
            case 8: return sgn ? scalar_type::i64 : scalar_type::u64;
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

extern "C" herr_t collect_link_cb(hid_t, const char* name, const H5L_info2_t*,
                                  void* op)
{
    static_cast<std::vector<std::string>*>(op)->emplace_back(name);
    return 0;
}

// Child link names of loc, name-ordered. nullopt = enumeration itself
// failed (torn metadata, T3 crashed-writer class).
std::optional<std::vector<std::string>> list_links(hid_t loc)
{
    std::vector<std::string> names;
    const herr_t rc = H5Literate2(loc, H5_INDEX_NAME, H5_ITER_INC, nullptr,
                                  collect_link_cb, &names);
    if (rc < 0) return std::nullopt;
    return names;
}

std::optional<H5O_type_t> object_type(hid_t loc, const char* name)
{
    H5O_info2_t info;
    if (H5Oget_info_by_name3(loc, name, &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0) {
        return std::nullopt;
    }
    return info.type;
}

// Opens and describes one dataset; on failure returns the reason string.
std::expected<dataset_desc, std::string>
describe_dataset(hid_t loc, const std::string& h5_path,
                 const std::string& display_name)
{
    hdf5::dataset ds = hdf5::dataset::open(loc, h5_path.c_str());
    if (!ds) return std::unexpected(harvest_h5_error("cannot open dataset"));

    const hid_t ftype = H5Dget_type(ds.id());
    if (ftype < 0) return std::unexpected(harvest_h5_error("cannot read dtype"));
    const std::optional<scalar_type> st = classify(ftype);
    H5Tclose(ftype);
    if (!st) return std::unexpected("unsupported dtype (not one of the 10 scalar writer types)");

    const hdf5::dataspace sp = hdf5::dataspace::from_dataset(ds.id());
    if (!sp) return std::unexpected(harvest_h5_error("cannot read dataspace"));
    const std::vector<hsize_t> dims = sp.dims();

    dataset_desc d;
    d.h5_path      = h5_path;
    d.display_name = display_name;
    d.dtype        = *st;
    if (dims.empty()) {
        d.rows      = 1; // scalar dataspace: a single value
        d.row_width = 1;
    } else {
        d.rows = dims[0];
        d.shape.assign(dims.begin() + 1, dims.end());
        d.row_width = std::accumulate(d.shape.begin(), d.shape.end(),
                                      std::uint64_t{1}, std::multiplies<>());
    }

    const hid_t dcpl = H5Dget_create_plist(ds.id());
    if (dcpl >= 0) {
        if (H5Pget_layout(dcpl) == H5D_CHUNKED) {
            std::vector<hsize_t> cdims(dims.empty() ? 1 : dims.size());
            const int n = H5Pget_chunk(dcpl, static_cast<int>(cdims.size()),
                                       cdims.data());
            if (n > 0) {
                d.chunk.assign(cdims.begin(), cdims.begin() + n);
            }
        }
        d.compressed = H5Pget_nfilters(dcpl) > 0;
        H5Pclose(dcpl);
    }
    return d;
}

tree_node meta_node(std::string name, std::string detail)
{
    tree_node n;
    n.name   = std::move(name);
    n.kind   = tree_node::kind_t::meta;
    n.detail = std::move(detail);
    return n;
}

tree_node& child_group(tree_node& parent, const std::string& name)
{
    for (tree_node& c : parent.children) {
        if (c.kind == tree_node::kind_t::group && c.name == name) return c;
    }
    tree_node g;
    g.name = name;
    g.kind = tree_node::kind_t::group;
    parent.children.push_back(std::move(g));
    return parent.children.back();
}

// Inserts a dataset leaf under root following '/'-separated segments of
// its display name (observe's semantic hierarchy).
void insert_dataset_path(tree_node& root, const std::string& display,
                         std::uint32_t dataset_index)
{
    tree_node*  cur = &root;
    std::size_t pos = 0;
    while (true) {
        const std::size_t next = display.find('/', pos);
        if (next == std::string::npos) break;
        cur = &child_group(*cur, display.substr(pos, next - pos));
        pos = next + 1;
    }
    tree_node leaf;
    leaf.name    = display.substr(pos);
    leaf.kind    = tree_node::kind_t::dataset;
    leaf.dataset = dataset_index;
    cur->children.push_back(std::move(leaf));
}

void add_dataset(file_model& m, tree_node& parent, dataset_desc desc)
{
    const auto idx = static_cast<std::uint32_t>(m.datasets.size());
    tree_node  leaf;
    leaf.name    = desc.display_name.substr(desc.display_name.rfind('/') + 1);
    leaf.kind    = tree_node::kind_t::dataset;
    leaf.dataset = idx;
    parent.children.push_back(std::move(leaf));
    m.datasets.push_back(std::move(desc));
}

// ---------------------------------------------------------------- observe

result<void> map_observe(hid_t fid, file_model& m)
{
    if (hdf5::group md = hdf5::group::open(fid, "metadata")) {
        if (auto v = hdf5::read_attr_string(md.id(), "created")) {
            m.metadata.emplace_back("created", *v);
        }
    }
    const auto names = list_links(fid);
    if (!names) {
        return std::unexpected(error{
            error_category::structure,
            harvest_h5_error("cannot enumerate root objects (torn metadata?)"),
            "/"});
    }
    constexpr std::string_view step_suffix = "__step_index";
    for (const std::string& name : *names) {
        if (name.ends_with(step_suffix)) continue;
        const auto otype = object_type(fid, name.c_str());
        if (!otype) {
            m.root.children.push_back(meta_node(
                name, harvest_h5_error("unreadable object")));
            continue;
        }
        if (*otype != H5O_TYPE_DATASET) continue; // observe roots hold only datasets

        std::string display = name;
        for (std::size_t p = 0; (p = display.find("__", p)) != std::string::npos;) {
            display.replace(p, 2, "/");
            ++p;
        }
        auto desc = describe_dataset(fid, "/" + name, display);
        if (!desc) {
            m.root.children.push_back(meta_node(display, desc.error()));
            continue;
        }
        const std::string sidx = name + std::string{step_suffix};
        if (std::ranges::find(*names, sidx) != names->end()) {
            desc->step_index_path = "/" + sidx;
        }
        const auto idx = static_cast<std::uint32_t>(m.datasets.size());
        m.datasets.push_back(std::move(*desc));
        insert_dataset_path(m.root, display, idx);
    }
    m.metadata.emplace_back("observe sites", std::to_string(m.datasets.size()));
    return {};
}

// -------------------------------------------------------------- buffer_log

result<void> map_buffer_log(hid_t fid, file_model& m)
{
    hdf5::group md = hdf5::group::open(fid, "metadata");
    if (!md) {
        return std::unexpected(error{error_category::structure,
                                     harvest_h5_error("cannot open /metadata"),
                                     "/metadata"});
    }
    if (auto v = hdf5::read_attr_string(md.id(), "schema_version")) {
        m.schema_version = *v;
    }
    tree_node meta_group;
    meta_group.name = "metadata";
    meta_group.kind = tree_node::kind_t::group;
    if (auto v = hdf5::read_attr_uint64(md.id(), "start_step")) {
        m.metadata.emplace_back("start_step", std::to_string(*v));
    }
    if (auto v = hdf5::read_attr_string(md.id(), "created")) {
        m.metadata.emplace_back("created", *v);
    }
    if (hdf5::attr_exists(md.id(), "config")) {
        m.metadata.emplace_back("embedded config", "yes");
    }

    // Topology: buffer id -> name map for semantic labels.
    std::vector<std::uint64_t> ids;
    std::vector<std::string>   names;
    if (hdf5::link_exists(md.id(), "topology")) {
        hdf5::group topo = hdf5::group::open(md.id(), "topology");
        if (topo) {
            // Labels only: a malformed table leaves the buffers unlabelled
            // rather than failing the browse.
            try {
                if (hdf5::dataset d = hdf5::dataset::open(topo.id(), "buffer_ids")) {
                    ids = hdf5::read_array<std::uint64_t>(d.id(), H5T_NATIVE_UINT64, 1).values;
                }
                if (hdf5::dataset d = hdf5::dataset::open(topo.id(), "buffer_names")) {
                    names = hdf5::read_fixed_strings(d.id());
                }
            } catch (const std::exception&) {
                ids.clear();
                names.clear();
            }
        }
    }
    auto name_of = [&](std::uint64_t id) -> std::string {
        for (std::size_t i = 0; i < ids.size() && i < names.size(); ++i) {
            if (ids[i] == id) return names[i];
        }
        return {};
    };

    // Steps group: step_numbers + timestamps as scannable datasets.
    tree_node steps_group;
    steps_group.name = "steps";
    steps_group.kind = tree_node::kind_t::group;
    if (hdf5::link_exists(fid, "steps")) {
        for (const char* n : {"step_numbers", "timestamps"}) {
            auto desc = describe_dataset(fid, std::string("/steps/") + n, n);
            if (desc) {
                add_dataset(m, steps_group, std::move(*desc));
            } else {
                steps_group.children.push_back(meta_node(n, desc.error()));
            }
        }
    }

    // Per-buffer data: /steps/buffers/<id>/data, numerically ordered.
    tree_node buffers_group;
    buffers_group.name = "buffers";
    buffers_group.kind = tree_node::kind_t::group;
    std::uint64_t n_buffers = 0;
    if (hdf5::link_exists(fid, "steps/buffers")) {
        hdf5::group bufs = hdf5::group::open(fid, "steps/buffers");
        auto        sub  = bufs ? list_links(bufs.id()) : std::nullopt;
        if (sub) {
            std::ranges::sort(*sub, [](const std::string& a, const std::string& b) {
                std::uint64_t ia = 0, ib = 0;
                std::from_chars(a.data(), a.data() + a.size(), ia);
                std::from_chars(b.data(), b.data() + b.size(), ib);
                return ia != ib ? ia < ib : a < b;
            });
            for (const std::string& id_str : *sub) {
                std::uint64_t id = 0;
                std::from_chars(id_str.data(), id_str.data() + id_str.size(), id);
                const std::string nm      = name_of(id);
                std::string       display = "buffer " + id_str;
                if (!nm.empty()) display += " (" + nm + ")";
                auto desc = describe_dataset(
                    fid, "/steps/buffers/" + id_str + "/data", display);
                if (desc) {
                    desc->step_index_path = "/steps/step_numbers";
                    add_dataset(m, buffers_group, std::move(*desc));
                    ++n_buffers;
                } else {
                    buffers_group.children.push_back(
                        meta_node(display, desc.error()));
                }
            }
        } else if (bufs) {
            buffers_group.children.push_back(meta_node(
                "buffers", harvest_h5_error("cannot enumerate buffers")));
        }
    }
    m.metadata.emplace_back("buffers", std::to_string(n_buffers));

    for (const auto& [k, v] : m.metadata) {
        meta_group.children.push_back(meta_node(k, v));
    }
    m.root.children.push_back(std::move(meta_group));
    m.root.children.push_back(std::move(steps_group));
    m.root.children.push_back(std::move(buffers_group));
    return {};
}

// -------------------------------------------------------------- map_worker

result<file_model> map_worker(const std::filesystem::path& path)
{
    const h5_error_silencer silence; // scoped to this job (see h5b_error.h)

    hdf5::file f = hdf5::file::open_read(path.string());
    if (!f) {
        return std::unexpected(
            error{error_category::open,
                  harvest_h5_error("cannot open file"), path.string()});
    }

    file_model m;
    m.family = file_family::unknown;
    m.path   = path;
    std::error_code ec;
    const auto      sz = std::filesystem::file_size(path, ec);
    m.file_bytes       = ec ? 0 : sz;
    m.root.name        = path.filename().string();
    m.root.kind        = tree_node::kind_t::root;

    // Every COAST log/state file carries a /metadata group stamped with a
    // `type` attribute. Read it uniformly; reject simulation-state checkpoints.
    if (!hdf5::link_exists(f.id(), "metadata")) {
        return std::unexpected(error{
            error_category::schema,
            "valid HDF5, but not a recognized COAST log file "
            "(observe / buffer log)",
            path.string()});
    }
    {
        hdf5::group md = hdf5::group::open(f.id(), "metadata");
        if (!md) {
            return std::unexpected(error{error_category::structure,
                                         harvest_h5_error("cannot open /metadata"),
                                         "/metadata"});
        }
        const std::optional<std::string> md_type = hdf5::read_attr_string(md.id(), "type");
        if (auto v = hdf5::read_attr_string(md.id(), "schema_version")) {
            m.schema_version = *v;
        }
        if (md_type == "simulation_state") {
            return std::unexpected(
                error{error_category::schema,
                      "simulation state checkpoint (--save-state), not a log file",
                      path.string()});
        }
        if (md_type == "observe_log") {
            m.family = file_family::observe;
        } else if (md_type == "buffer_log") {
            m.family = file_family::buffer_log;
        }
    }

    if (m.family == file_family::unknown) {
        return std::unexpected(error{
            error_category::schema,
            "valid HDF5, but not a recognized COAST log file "
            "(observe / buffer log)",
            path.string()});
    }

    m.metadata.emplace_back("family", file_family_name(m.family));
    if (!m.schema_version.empty()) {
        m.metadata.emplace_back("schema_version", m.schema_version);
    }

    result<void> mapped;
    switch (m.family) {
        case file_family::observe:    mapped = map_observe(f.id(), m); break;
        case file_family::buffer_log: mapped = map_buffer_log(f.id(), m); break;
        case file_family::unknown:    break; // unreachable
    }
    if (!mapped) return std::unexpected(mapped.error());

    if (m.datasets.empty()) {
        return std::unexpected(error{error_category::structure,
                                     std::format("{} file contains no readable datasets",
                                                 file_family_name(m.family)),
                                     path.string()});
    }
    return m;
}

} // namespace

result<file_model> open_and_map(hdf5::executor&              exec,
                                const std::filesystem::path& path)
{
    // Cheap pre-checks on the calling thread — pure std::filesystem.
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return std::unexpected(error{error_category::open,
                                     "file does not exist", path.string()});
    }

    return exec.submit_sync([&]() -> result<file_model> {
        try {
            return map_worker(path);
        } catch (const std::exception& e) {
            // The hdf5::read_attr_* helpers throw on a malformed attribute
            // (more than one element, a variable-length or non-string value);
            // nothing may escape the job.
            return std::unexpected(error{
                error_category::open,
                std::string{"unexpected exception during open: "} + e.what(),
                path.string()});
        }
    });
}

} // namespace h5b
