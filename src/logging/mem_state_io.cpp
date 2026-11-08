//
// In-memory state tree + adapters — implementation.
//

#include "logging/mem_state_io.h"
#include "hdf5/native_type.h"
#include "logging/hdf5_utils.h"   // hdf5::file + direct libhdf5 (parse path only)
#include "hdf5_parse.h"

#include <array>
#include <cstring>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace logging {

namespace {

// Trait: is U a std::vector<E>?
template <class>            struct is_std_vector              : std::false_type {};
template <class E>          struct is_std_vector<std::vector<E>> : std::true_type {};

} // namespace

// --- mem_state_writer -------------------------------------------------------

void mem_state_writer::begin_group(std::string_view name)
{
    // try_emplace keeps an existing group of the same name (matching how a
    // writer would reopen a path); the node address is stable, so pushing it is
    // safe even as sibling groups are added later.
    auto [it, _] = cur().groups.try_emplace(std::string(name));
    stack_.push_back(&it->second);
}

void mem_state_writer::end_group()
{
    if (stack_.size() > 1) stack_.pop_back();
}

void mem_state_writer::write(std::string_view name, double value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, float value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, std::int32_t value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, std::int64_t value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, std::uint32_t value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, std::uint64_t value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}
void mem_state_writer::write(std::string_view name, bool value)
{
    cur().fields.insert_or_assign(std::string(name), mem_value(value));
}

void mem_state_writer::write(std::string_view name, std::span<const float> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<float>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const double> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<double>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const std::int32_t> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<std::int32_t>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const std::uint32_t> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<std::uint32_t>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const std::uint64_t> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<std::uint64_t>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const std::uint16_t> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<std::uint16_t>(values.begin(), values.end())));
}
void mem_state_writer::write(std::string_view name, std::span<const std::uint8_t> values)
{
    cur().fields.insert_or_assign(std::string(name),
                                  mem_value(std::vector<std::uint8_t>(values.begin(), values.end())));
}

void mem_state_writer::write_2d(std::string_view name, std::span<const float> values,
                                std::size_t rows, std::size_t cols)
{
    cur().fields.insert_or_assign(
        std::string(name),
        mem_value(mem_array2d{std::vector<float>(values.begin(), values.end()), rows, cols}));
}

void mem_state_writer::write_attr(std::string_view name, std::string_view value)
{
    cur().attrs.insert_or_assign(std::string(name), std::string(value));
}

// --- mem_state_reader -------------------------------------------------------

const mem_value* mem_state_reader::find_field(std::string_view name) const
{
    const auto& f = cur().fields;
    if (auto it = f.find(name); it != f.end()) return &it->second;
    // hdf5_state_reader opens the name as a dataset, which fails on a group.
    if (cur().groups.contains(name)) {
        throw std::runtime_error("state field " + stack_.back().path + "/" + std::string(name) +
                                 ": is a group, not a dataset");
    }
    return nullptr;
}

namespace {

int rank_of(const mem_value& v)
{
    return std::visit(
        [](const auto& x) {
            using U = std::decay_t<decltype(x)>;
            if constexpr (std::is_arithmetic_v<U>) return 0;
            else if constexpr (is_std_vector<U>::value) return 1;
            else return 2;
        },
        v);
}

} // namespace

void mem_state_reader::wrong_rank(std::string_view name, int expected, const mem_value& v) const
{
    throw std::runtime_error("state field " + stack_.back().path + "/" + std::string(name) +
                             ": expected rank " + std::to_string(expected) +
                             ", the file has rank " + std::to_string(rank_of(v)));
}

bool mem_state_reader::enter_group(std::string_view name)
{
    const auto& g = cur().groups;
    auto it = g.find(name);
    if (it == g.end()) return false;
    stack_.push_back({&it->second, stack_.back().path + "/" + std::string(name)});
    return true;
}

void mem_state_reader::leave_group()
{
    if (stack_.size() > 1) stack_.pop_back();
}

bool mem_state_reader::has_field(std::string_view name)
{
    // Mirror hdf5_state_reader::has_field == H5Lexists (groups + datasets).
    return cur().fields.contains(name) || cur().groups.contains(name);
}

// Reads convert element types the way libhdf5's native conversion does
// (numeric static_cast); a rank other than the read's throws (wrong_rank), as
// hdf5_state_reader's require_rank does.

template <class T>
T mem_state_reader::scalar(std::string_view name, T default_value) const
{
    const mem_value* v = find_field(name);
    if (!v) return default_value;
    return std::visit(
        [&](const auto& x) -> T {
            using U = std::decay_t<decltype(x)>;
            if constexpr (std::is_arithmetic_v<U>) return static_cast<T>(x);
            else wrong_rank(name, 0, *v);
        },
        *v);
}

template <class T>
std::vector<T> mem_state_reader::vec(std::string_view name) const
{
    const mem_value* v = find_field(name);
    if (!v) return {};
    return std::visit(
        [&](const auto& x) -> std::vector<T> {
            using U = std::decay_t<decltype(x)>;
            if constexpr (is_std_vector<U>::value) {
                std::vector<T> out(x.size());
                for (std::size_t i = 0; i < x.size(); ++i) out[i] = static_cast<T>(x[i]);
                return out;
            } else {
                wrong_rank(name, 1, *v);
            }
        },
        *v);
}

double mem_state_reader::read_double(std::string_view name, double default_value)
{ return scalar<double>(name, default_value); }

float mem_state_reader::read_float(std::string_view name, float default_value)
{ return scalar<float>(name, default_value); }

std::int32_t mem_state_reader::read_int32(std::string_view name, std::int32_t default_value)
{ return scalar<std::int32_t>(name, default_value); }

std::int64_t mem_state_reader::read_int64(std::string_view name, std::int64_t default_value)
{ return scalar<std::int64_t>(name, default_value); }

std::uint32_t mem_state_reader::read_uint32(std::string_view name, std::uint32_t default_value)
{ return scalar<std::uint32_t>(name, default_value); }

std::uint64_t mem_state_reader::read_uint64(std::string_view name, std::uint64_t default_value)
{ return scalar<std::uint64_t>(name, default_value); }

bool mem_state_reader::read_bool(std::string_view name, bool default_value)
{
    // hdf5_state_reader stores bool as int8 and returns (value != 0). Reading
    // as a wider integer then comparing preserves that for any stored scalar.
    if (!find_field(name)) return default_value;
    return scalar<std::int64_t>(name, 0) != 0;
}

std::vector<float> mem_state_reader::read_float_vec(std::string_view name)
{ return vec<float>(name); }

std::vector<double> mem_state_reader::read_double_vec(std::string_view name)
{ return vec<double>(name); }

std::vector<std::int32_t> mem_state_reader::read_int32_vec(std::string_view name)
{ return vec<std::int32_t>(name); }

std::vector<std::uint32_t> mem_state_reader::read_uint32_vec(std::string_view name)
{ return vec<std::uint32_t>(name); }

std::vector<std::uint64_t> mem_state_reader::read_uint64_vec(std::string_view name)
{ return vec<std::uint64_t>(name); }

std::vector<std::uint16_t> mem_state_reader::read_uint16_vec(std::string_view name)
{ return vec<std::uint16_t>(name); }

std::vector<std::uint8_t> mem_state_reader::read_uint8_vec(std::string_view name)
{ return vec<std::uint8_t>(name); }

std::vector<float> mem_state_reader::read_float_2d(std::string_view name, array_2d_info& info)
{
    info = {};
    const mem_value* v = find_field(name);
    if (!v) return {};
    const auto* a = std::get_if<mem_array2d>(v);
    if (!a) wrong_rank(name, 2, *v);
    info.rows = a->rows;
    info.cols = a->cols;
    return a->data;
}

std::string mem_state_reader::read_attr(std::string_view name)
{
    const auto& a = cur().attrs;
    auto it = a.find(name);
    return it == a.end() ? std::string{} : it->second;
}

std::vector<std::pair<std::string, scalar_value>> mem_state_reader::read_scalars()
{
    return detail::scalars_of(cur());
}

// --- replay (tree -> any writer) --------------------------------------------

void replay(const mem_group& tree, state_writer& writer)
{
    // Attrs and datasets live in independent HDF5 namespaces within a group, so
    // emitting attrs-then-fields-then-subgroups reproduces the same structure as
    // any other ordering. Names are unique within each map.
    for (const auto& [name, value] : tree.attrs) {
        writer.write_attr(name, value);
    }
    for (const auto& [name, value] : tree.fields) {
        std::visit(
            [&](const auto& x) {
                using U = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<U, mem_array2d>) {
                    writer.write_2d(name, std::span<const float>(x.data), x.rows, x.cols);
                } else if constexpr (is_std_vector<U>::value) {
                    writer.write(name, std::span<const typename U::value_type>(x));
                } else {
                    writer.write(name, x);   // scalar overload picked by type
                }
            },
            value);
    }
    for (const auto& [name, sub] : tree.groups) {
        writer.begin_group(name);
        replay(sub, writer);
        writer.end_group();
    }
}

// --- parse (HDF5 -> tree) ---------------------------------------------------
//
// Generic enumeration of an HDF5 file into the tree. The fiddly part is dtype
// detection: we recover element type and shape from each dataset's datatype
// (class/size/sign) and dataspace (rank) and pick the matching mem_value
// alternative so mem_state_reader hands back the original type. The producible
// set is exactly the state_io interface — the mapping below is exhaustive over
// it; another integer or float width falls back to the closest reading.
//
// The same walk checks a file before any of it is applied (both load paths
// parse it first, state_logger.cpp parse_open_file), so both refuse the same
// files. A state file holds only what state_writer produces:
//   - datasets of rank 0, 1 or 2 with an integer or float type, not H5S_NULL,
//     whose declared size the file stores (hdf5::require_stored);
//   - attributes holding one fixed-length string;
//   - hard links to a group or a dataset, each object reached by one link
//     only (a tree), at most k_max_depth groups deep.
// Anything else throws, naming the object, whether or not a load would read
// it. The tree rule matters for the walk itself: a link back to an ancestor
// would recurse until the stack overflows, groups shared between links would
// be walked exponentially often, and a soft or external link could point it
// anywhere, including another file. Whether a field has the rank its reader
// expects is left to the read.

namespace {

struct field_type {
    int         rank;
    H5T_class_t cls;
    std::size_t size;
    H5T_sign_t  sign;   // H5T_SGN_ERROR unless an integer
};

field_type check_dataset(hid_t ds)
{
    const int rank = [&] {
        const hdf5::dataspace space = hdf5::dataspace::from_dataset(ds);
        return space ? H5Sget_simple_extent_ndims(space.id()) : -1;
    }();
    if (rank < 0 || rank > 2) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(ds) + ": rank " + std::to_string(rank) +
                                 ", a state file holds rank 0, 1 or 2");
    }
    // require_rank refuses H5S_NULL; require_stored a size the file does not
    // store (#159), here so that startup's check refuses what the runtime
    // parse would.
    hdf5::require_stored(ds, hdf5::element_count(ds, hdf5::require_rank(ds, rank)));

    const hdf5::id_guard dtype(H5Dget_type(ds), H5Tclose);
    const H5T_class_t    cls = H5Tget_class(dtype.id());
    if (cls != H5T_INTEGER && cls != H5T_FLOAT) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(ds) +
                                 ": not an integer or float dataset");
    }
    return {rank, cls, H5Tget_size(dtype.id()),
            cls == H5T_INTEGER ? H5Tget_sign(dtype.id()) : H5T_SGN_ERROR};
}

mem_value read_dataset_value(hid_t ds, const field_type& t)
{
    const auto [rank, cls, sz, sgn] = t;
    // The element type picks its HDF5 memory type (hdf5::native_type).
    auto read = [&](auto proto) {
        using T = decltype(proto);
        return hdf5::read_array<T>(ds, hdf5::native_type<T>(), rank);
    };

    if (rank == 0) {                         // scalar
        // The terminal else handles f64 AND any out-of-set scalar (best-effort
        // f64) in one branch — no duplicated body.
        if (cls == H5T_FLOAT && sz == 4)               return read(float{}).values[0];
        if (cls == H5T_INTEGER && sz == 1)             return read(std::int8_t{}).values[0] != 0;
        if (cls == H5T_INTEGER && sz == 4 && sgn == H5T_SGN_2) return read(std::int32_t{}).values[0];
        if (cls == H5T_INTEGER && sz == 4)             return read(std::uint32_t{}).values[0];
        if (cls == H5T_INTEGER && sz == 8 && sgn == H5T_SGN_2) return read(std::int64_t{}).values[0];
        if (cls == H5T_INTEGER && sz == 8)             return read(std::uint64_t{}).values[0];
        return read(double{}).values[0];
    }
    if (rank == 1) {                         // vector
        // The terminal else handles f32 AND any out-of-set vector (best-effort
        // f32) in one branch — no duplicated body.
        if (cls == H5T_FLOAT && sz == 8)               return read(double{}).values;
        if (cls == H5T_INTEGER && sz == 8)             return read(std::uint64_t{}).values;
        if (cls == H5T_INTEGER && sz == 4 && sgn == H5T_SGN_2) return read(std::int32_t{}).values;
        if (cls == H5T_INTEGER && sz == 4)             return read(std::uint32_t{}).values;
        if (cls == H5T_INTEGER && sz == 2)             return read(std::uint16_t{}).values;
        if (cls == H5T_INTEGER && sz == 1)             return read(std::uint8_t{}).values;
        return read(float{}).values;
    }
    // 2D (state_io produces only 2D-f32)
    auto a = read(float{});
    return mem_array2d{std::move(a.values), a.dims[0], a.dims[1]};
}

// With `out` null, only checks.
void parse_attrs(hid_t loc, mem_group* out)
{
    H5O_info2_t oi{};
    if (H5Oget_info3(loc, &oi, H5O_INFO_NUM_ATTRS) < 0) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(loc) + ": cannot list its attributes");
    }

    for (hsize_t i = 0; i < oi.num_attrs; ++i) {
        const hdf5::id_guard a(H5Aopen_by_idx(loc, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                                              H5P_DEFAULT, H5P_DEFAULT),
                               H5Aclose);
        if (a.id() < 0) {
            throw std::runtime_error("hdf5: " + hdf5::object_path(loc) + ": attribute " +
                                     std::to_string(i) + " cannot be opened");
        }

        // Through state_writer, every attribute is one fixed-length string;
        // read_string_attr refuses anything else.
        std::string value = hdf5::read_string_attr(a.id());
        if (!out) continue;

        const ssize_t nlen = H5Aget_name(a.id(), 0, nullptr);
        std::string   name(nlen > 0 ? static_cast<std::size_t>(nlen) : 0, '\0');
        if (nlen > 0) H5Aget_name(a.id(), static_cast<std::size_t>(nlen) + 1, name.data());
        out->attrs.insert_or_assign(std::move(name), std::move(value));
    }
}

// The deepest object a checkpoint holds today is 8 groups down (ngm2
// dendrites); the cap only has to keep the recursion off the stack limit.
constexpr int k_max_depth = 64;

// Objects reached so far, by address token, so none is walked twice.
class object_set {
    using key = std::array<unsigned char, sizeof(H5O_token_t)>;
    std::set<key> seen_;

public:
    // False if `token` was seen before. Tokens of one file compare by bytes.
    bool insert(const H5O_token_t& token)
    {
        key k;
        std::memcpy(k.data(), &token, k.size());
        return seen_.insert(k).second;
    }
};

void parse_group_at(hid_t gid, mem_group* out, bool recurse, object_set& seen, int depth)
{
    parse_attrs(gid, out);

    H5G_info_t gi{};
    if (H5Gget_info(gid, &gi) < 0) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(gid) + ": cannot list its members");
    }

    for (hsize_t i = 0; i < gi.nlinks; ++i) {
        const ssize_t nlen = H5Lget_name_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC,
                                                i, nullptr, 0, H5P_DEFAULT);
        if (nlen < 0) {
            throw std::runtime_error("hdf5: " + hdf5::object_path(gid) + ": link " +
                                     std::to_string(i) + " has no readable name");
        }
        std::string name(static_cast<std::size_t>(nlen), '\0');
        H5Lget_name_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, i,
                           name.data(), static_cast<std::size_t>(nlen) + 1, H5P_DEFAULT);
        auto where = [&] { return "hdf5: " + hdf5::object_path(gid) + "/" + name; };

        H5L_info2_t li{};
        if (H5Lget_info2(gid, name.c_str(), &li, H5P_DEFAULT) < 0) {
            throw std::runtime_error(where() + ": link cannot be inspected");
        }
        if (li.type != H5L_TYPE_HARD) {
            throw std::runtime_error(where() + ": a soft or external link, a state file holds "
                                               "hard links only");
        }

        H5O_info2_t oi{};
        if (H5Oget_info_by_name3(gid, name.c_str(), &oi, H5O_INFO_BASIC, H5P_DEFAULT) < 0) {
            throw std::runtime_error(where() + ": link does not resolve to an object");
        }
        if (oi.type == H5O_TYPE_GROUP && !recurse) continue;
        if (!seen.insert(oi.token)) {
            throw std::runtime_error(where() + ": links to an object reached before (a cycle "
                                               "or a shared object)");
        }

        if (oi.type == H5O_TYPE_GROUP) {
            if (depth >= k_max_depth) {
                throw std::runtime_error(where() + ": nested deeper than " +
                                         std::to_string(k_max_depth) + " groups");
            }
            const auto sub = hdf5::group::open(gid, name.c_str());
            if (!sub) throw std::runtime_error(where() + ": group cannot be opened");
            parse_group_at(sub.id(), out ? &out->groups[name] : nullptr, recurse, seen, depth + 1);
        } else if (oi.type == H5O_TYPE_DATASET) {
            const auto ds = hdf5::dataset::open(gid, name.c_str());
            if (!ds) throw std::runtime_error(where() + ": dataset cannot be opened");
            const field_type t = check_dataset(ds.id());
            if (out) out->fields.insert_or_assign(name, read_dataset_value(ds.id(), t));
        } else {
            throw std::runtime_error(where() + ": neither a group nor a dataset");
        }
    }
}

} // namespace

namespace detail {

void parse_group(hid_t gid, mem_group* out, bool recurse)
{
    object_set  seen;
    H5O_info2_t oi{};
    if (H5Oget_info3(gid, &oi, H5O_INFO_BASIC) < 0) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(gid) + ": cannot inspect it");
    }
    seen.insert(oi.token);
    parse_group_at(gid, out, recurse, seen, 0);
}

std::vector<std::pair<std::string, scalar_value>> scalars_of(const mem_group& g)
{
    std::vector<std::pair<std::string, scalar_value>> out;
    for (const auto& [name, value] : g.fields)
        std::visit(
            [&](const auto& x) {
                using U = std::decay_t<decltype(x)>;
                if constexpr (std::is_arithmetic_v<U>)   // the scalar alternatives
                    out.emplace_back(name, scalar_value{ std::in_place_type<U>, x });
            },
            value);
    for (const auto& [name, value] : g.attrs) out.emplace_back(name, scalar_value{ value });
    return out;
}

} // namespace detail

mem_group parse_hdf5_to_tree(hdf5::file& file)
{
    mem_group root;
    if (file) detail::parse_group(file.id(), &root, /*recurse=*/true);
    return root;
}

} // namespace logging
