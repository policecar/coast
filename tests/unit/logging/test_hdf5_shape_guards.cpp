// tests/unit/logging/test_hdf5_shape_guards.cpp
//
// Checkpoint readers refuse a dataset or attribute whose shape or type the
// writer never produces (#150). Before the fix, H5Dread(H5S_ALL) and H5Aread
// transferred as many elements as the FILE declared into a buffer sized from
// part of the shape — a heap or stack overflow with file-chosen length and
// content (run under -DCOAST_SANITIZER=asan to see it on the old code).
//
// Both load paths (startup load_state, runtime parse_state + apply_state)
// refuse the same files:
//   - an object a state writer never produces, read by a load or not, before
//     any entity is loaded (parse_hdf5_to_tree, which both load paths run first);
//   - a field with another rank than its reader expects, when it is read.
// Errors name the object, and a refusal leaves no HDF5 object open.
//
// Each case writes a valid checkpoint, then replaces one object in it with a
// crafted one through raw libhdf5 on the executor thread.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"
#include "logging/mem_state_io.h"
#include "logging/state_io.h"
#include "logging/state_logger.h"
#include "sim_core/sim_env.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using logging::state_logger;

namespace {

// An entity whose state has one field of each reader shape: a vector, a
// scalar and a 2-D array. Counts its load_state calls.
struct shaped_entity : sim::io_entity {
    sim::output_id_t oid;
    int&             loads;

    shaped_entity(sim::output_id_t id, int& n) : oid(id), loads(n) {}

    sim::step_result process() override { return {}; }
    void save_state(logging::state_writer& w) const override
    {
        const std::array<float, 3> v{1.f, 2.f, 3.f};
        const std::array<float, 4> m{1.f, 2.f, 3.f, 4.f};
        w.write("v", std::span<const float>(v));
        w.write("s", 0.5);
        w.write_2d("m", std::span<const float>(m), 2, 2);
    }
    void load_state(logging::state_reader& r) override
    {
        ++loads;
        logging::state_reader::array_2d_info info;
        (void)r.read_float_vec("v");
        (void)r.read_double("s", 0.0);
        (void)r.read_float_2d("m", info);
    }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

struct fixture {
    int            loads = 0;
    sim::env       env;
    hdf5::executor h5;
    fs::path       path = fs::temp_directory_path() /
                          ("coast_shape_guards_" + std::to_string(::getpid()) + ".h5");

    fixture()
    {
        env.emplace_back<shaped_entity>(sim::output_id_t{4}, loads);
        logging::state_save_config cfg;
        cfg.output_path = path;
        REQUIRE(state_logger::serialize_state(state_logger::extract_state(env, 7), cfg, h5)
                    .get().success);
    }
    ~fixture() { fs::remove(path); }

    // Runs `fn` on the open file, on the executor thread.
    void edit(const std::function<void(hid_t)>& fn)
    {
        h5.submit_sync([&] {
            const hid_t f = H5Fopen(path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            REQUIRE(f >= 0);
            fn(f);
            H5Fclose(f);
        });
    }

    // Replaces dataset `name` with one of `type` over dataspace `space`, filled
    // with `n` bytes of 0x01 (or left unwritten when n == 0).
    void replace_dataset(const char* name, hid_t type, hid_t space, std::size_t n)
    {
        edit([&](hid_t f) {
            REQUIRE(H5Ldelete(f, name, H5P_DEFAULT) >= 0);
            const hid_t ds = H5Dcreate2(f, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(ds >= 0);
            if (n > 0) {
                const std::vector<unsigned char> bytes(n, 1);
                REQUIRE(H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data()) >= 0);
            }
            H5Dclose(ds);
        });
    }

    // No HDF5 object of any file is open (a refusal leaked none).
    void check_nothing_open()
    {
        h5.submit_sync([] { CHECK(H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL) == 0); });
    }

    // Both load paths refuse the file before any entity is loaded; the message
    // names `what`.
    void check_refused(const std::string& what)
    {
        const auto startup = state_logger{}.load_state(env, path, h5);
        CHECK_FALSE(startup.success);
        CHECK_MESSAGE(startup.error_message.find(what) != std::string::npos, startup.error_message);
        CHECK(loads == 0);
        check_nothing_open();

        const auto pr = state_logger::parse_state(path, h5);
        CHECK_FALSE(pr.result.success);
        CHECK_MESSAGE(pr.result.error_message.find(what) != std::string::npos, pr.result.error_message);
        check_nothing_open();
    }

    // Both paths refuse the file when the field is read: startup's load fails,
    // the runtime parse succeeds and its apply throws. The message names `what`.
    void check_refused_on_read(const std::string& what)
    {
        const auto startup = state_logger{}.load_state(env, path, h5);
        CHECK_FALSE(startup.success);
        CHECK_MESSAGE(startup.error_message.find(what) != std::string::npos, startup.error_message);
        check_nothing_open();

        const auto pr = state_logger::parse_state(path, h5);
        REQUIRE_MESSAGE(pr.result.success, pr.result.error_message);
        CHECK_THROWS_WITH_AS((void)state_logger::apply_state(env, pr.tree), doctest::Contains(what.c_str()),
                             std::runtime_error);
    }

    // Adds a scalar attribute of `type` holding `value` to group `group`.
    void add_attr(const char* group, const char* name, hid_t type, const void* value)
    {
        edit([&](hid_t f) {
            const hid_t g  = H5Gopen2(f, group, H5P_DEFAULT);
            const hid_t sp = H5Screate(H5S_SCALAR);
            const hid_t a  = H5Acreate2(g, name, type, sp, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(a >= 0);
            REQUIRE(H5Awrite(a, type, value) >= 0);
            H5Aclose(a);
            H5Sclose(sp);
            H5Gclose(g);
        });
    }
};

// Simple dataspace of the given dims (the caller closes it).
hid_t simple_space(std::initializer_list<hsize_t> dims)
{
    const std::vector<hsize_t> d(dims);
    return H5Screate_simple(static_cast<int>(d.size()), d.data(), nullptr);
}

} // namespace

TEST_CASE("an untouched checkpoint loads on both paths")
{
    fixture f;
    const auto startup = state_logger{}.load_state(f.env, f.path, f.h5);
    CHECK_MESSAGE(startup.success, startup.error_message);
    CHECK(f.loads == 1);

    const auto pr = state_logger::parse_state(f.path, f.h5);
    REQUIRE_MESSAGE(pr.result.success, pr.result.error_message);
    CHECK(state_logger::apply_state(f.env, pr.tree).success);
    CHECK(f.loads == 2);
}

TEST_CASE("a rank-2 dataset where a vector is expected is refused")
{
    fixture f;
    const hid_t sp = simple_space({16, 16});
    f.replace_dataset("/entities/4/v", H5T_NATIVE_FLOAT, sp, 16 * 16 * sizeof(float));
    H5Sclose(sp);
    // A 2-D array is a valid state field, so only the read finds it wrong.
    f.check_refused_on_read("/entities/4/v: expected rank 1, the file has rank 2");
}

TEST_CASE("a rank-0 dataset where a vector is expected is refused and not read as empty")
{
    fixture f;
    const hid_t sp = H5Screate(H5S_SCALAR);
    f.replace_dataset("/entities/4/v", H5T_NATIVE_FLOAT, sp, sizeof(float));
    H5Sclose(sp);
    f.check_refused_on_read("/entities/4/v: expected rank 1, the file has rank 0");
}

TEST_CASE("a multi-element dataset where a scalar is expected is refused")
{
    fixture f;
    const hid_t sp = simple_space({64});
    f.replace_dataset("/entities/4/s", H5T_NATIVE_DOUBLE, sp, 64 * sizeof(double));
    H5Sclose(sp);
    f.check_refused_on_read("/entities/4/s: expected rank 0, the file has rank 1");
}

TEST_CASE("a null dataspace where a scalar is expected is refused on both paths")
{
    // H5S_NULL reports rank 0, and H5Dread from it succeeds without writing:
    // a rank-only check would hand back an uninitialized value.
    fixture f;
    const hid_t sp = H5Screate(H5S_NULL);
    f.replace_dataset("/entities/4/s", H5T_NATIVE_DOUBLE, sp, 0);
    H5Sclose(sp);
    f.check_refused("null dataspace");
}

TEST_CASE("a string dataset where a number is expected is refused on both paths")
{
    fixture f;
    const hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, 8);
    const hid_t sp = H5Screate(H5S_SCALAR);
    f.replace_dataset("/entities/4/s", st, sp, 8);
    H5Sclose(sp);
    H5Tclose(st);
    f.check_refused("/entities/4/s: not an integer or float dataset");
}

TEST_CASE("a rank-3 dataset is refused on both paths")
{
    // Runtime: read_dataset_value used to query dims into hsize_t[2] — a stack
    // overflow by one hsize_t per extra dimension.
    fixture f;
    const hid_t sp = simple_space({2, 2, 2});
    f.replace_dataset("/entities/4/m", H5T_NATIVE_FLOAT, sp, 8 * sizeof(float));
    H5Sclose(sp);
    f.check_refused("/entities/4/m");
}

// A field's dims are metadata, so a small file can declare a dataset far larger
// than it stores (#159). Each case pins one term of hdf5::require_stored: the
// contiguous bound, the chunked bound (allocated chunks, not dims), and the
// uncompressed chunk size (a compressed dataset stores less than it declares).

TEST_CASE("a contiguous dataset that declares more than the file stores is refused on both paths")
{
    // [2^34] floats, never written: contiguous storage is allocated late, so
    // the file stores none of the 64 GiB a reader would have allocated.
    fixture f;
    const hid_t sp = simple_space({hsize_t{1} << 34});
    f.replace_dataset("/entities/4/v", H5T_NATIVE_FLOAT, sp, 0);
    H5Sclose(sp);
    f.check_refused("/entities/4/v: declares 68719476736 bytes, the file stores at most 0");
}

TEST_CASE("a chunked dataset is bounded by its allocated chunks, not its dims")
{
    // [2^28] floats in chunks of 1024, only the first chunk written: 1 GiB
    // declared, 4 KiB stored.
    fixture f;
    f.edit([](hid_t file) {
        REQUIRE(H5Ldelete(file, "/entities/4/v", H5P_DEFAULT) >= 0);
        const hid_t   sp    = simple_space({hsize_t{1} << 28});
        const hid_t   dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t chunk = 1024;
        REQUIRE(H5Pset_chunk(dcpl, 1, &chunk) >= 0);
        const hid_t ds = H5Dcreate2(file, "/entities/4/v", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        REQUIRE(ds >= 0);
        const hsize_t            start = 0;
        const hid_t              mem   = simple_space({chunk});
        const std::vector<float> ones(chunk, 1.f);
        REQUIRE(H5Sselect_hyperslab(sp, H5S_SELECT_SET, &start, nullptr, &chunk, nullptr) >= 0);
        REQUIRE(H5Dwrite(ds, H5T_NATIVE_FLOAT, mem, sp, H5P_DEFAULT, ones.data()) >= 0);
        for (const hid_t id : {mem, sp}) H5Sclose(id);
        H5Dclose(ds);
        H5Pclose(dcpl);
    });
    f.check_refused("/entities/4/v: declares 1073741824 bytes, the file stores at most 4096");
}

TEST_CASE("a fully written compressed dataset loads on both paths, though it stores less than it declares")
{
    // 2^16 zero floats, deflated: every chunk is allocated, each far smaller
    // than the 16 KiB it expands to.
    constexpr hsize_t n = hsize_t{1} << 16;
    fixture           f;
    f.edit([&](hid_t file) {
        REQUIRE(H5Ldelete(file, "/entities/4/v", H5P_DEFAULT) >= 0);
        const hid_t   sp    = simple_space({n});
        const hid_t   dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t chunk = 4096;
        REQUIRE(H5Pset_chunk(dcpl, 1, &chunk) >= 0);
        REQUIRE(H5Pset_deflate(dcpl, 9) >= 0);
        const hid_t ds = H5Dcreate2(file, "/entities/4/v", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        REQUIRE(ds >= 0);
        const std::vector<float> zeros(n, 0.f);
        REQUIRE(H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, zeros.data()) >= 0);
        H5Dclose(ds);
        H5Sclose(sp);
        H5Pclose(dcpl);
    });
    // The case only tests the uncompressed bound if the storage is smaller.
    f.h5.submit_sync([&] {
        const hid_t file = H5Fopen(f.path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        const hid_t ds   = H5Dopen2(file, "/entities/4/v", H5P_DEFAULT);
        CHECK(H5Dget_storage_size(ds) < n * sizeof(float) / 10);
        H5Dclose(ds);
        H5Fclose(file);
    });

    const auto startup = state_logger{}.load_state(f.env, f.path, f.h5);
    CHECK_MESSAGE(startup.success, startup.error_message);
    const auto pr = state_logger::parse_state(f.path, f.h5);
    REQUIRE_MESSAGE(pr.result.success, pr.result.error_message);
    CHECK(state_logger::apply_state(f.env, pr.tree).success);
    CHECK(f.loads == 2);
}

TEST_CASE("a rank-1 metadata/step_count is refused by parse_state, not thrown")
{
    // A rank-1 dataset is a valid state field, so the whole-file check passes
    // it; only the read finds it wrong. parse_state read step_count after its
    // try, so the throw escaped through load_state into the GUI and
    // terminated the process.
    fixture f;
    const hid_t sp = simple_space({4});
    f.replace_dataset("/metadata/step_count", H5T_NATIVE_UINT64, sp, 4 * sizeof(std::uint64_t));
    H5Sclose(sp);
    f.check_refused("/metadata/step_count");
}

TEST_CASE("a two-element schema_version attribute is refused on both paths")
{
    fixture f;
    f.edit([](hid_t file) {
        const hid_t md = H5Gopen2(file, "metadata", H5P_DEFAULT);
        REQUIRE(H5Adelete(md, "schema_version") >= 0);
        // 16-byte strings: past std::string's inline buffer, so the old
        // one-element read overflowed a heap allocation ASan can see.
        const hid_t st = H5Tcopy(H5T_C_S1);
        H5Tset_size(st, 16);
        const hid_t sp = simple_space({2});
        const hid_t a  = H5Acreate2(md, "schema_version", st, sp, H5P_DEFAULT, H5P_DEFAULT);
        const std::string two(32, '\0');
        REQUIRE(H5Awrite(a, st, two.data()) >= 0);
        H5Aclose(a);
        H5Sclose(sp);
        H5Tclose(st);
        H5Gclose(md);
    });
    f.check_refused("attribute 'schema_version': holds 2 elements, expected 1");
}

TEST_CASE("a variable-length string attribute is refused on both paths")
{
    fixture f;
    f.edit([](hid_t file) {
        const hid_t md = H5Gopen2(file, "metadata", H5P_DEFAULT);
        REQUIRE(H5Adelete(md, "schema_version") >= 0);
        const hid_t st = H5Tcopy(H5T_C_S1);
        H5Tset_size(st, H5T_VARIABLE);
        const hid_t sp = H5Screate(H5S_SCALAR);
        const hid_t a  = H5Acreate2(md, "schema_version", st, sp, H5P_DEFAULT, H5P_DEFAULT);
        const char* v  = "1";
        REQUIRE(H5Awrite(a, st, &v) >= 0);
        H5Aclose(a);
        H5Sclose(sp);
        H5Tclose(st);
        H5Gclose(md);
    });
    f.check_refused("variable-length string");
}

TEST_CASE("objects no load reads are checked too, so both paths refuse the same files")
{
    SUBCASE("a variable-length string attribute (h5py's default)")
    {
        fixture f;
        const hid_t st = H5Tcopy(H5T_C_S1);
        H5Tset_size(st, H5T_VARIABLE);
        const char* v = "baseline";
        f.add_attr("entities", "note", st, &v);
        H5Tclose(st);
        f.check_refused("/entities attribute 'note': is a variable-length string");
    }
    SUBCASE("a non-string attribute")
    {
        fixture f;
        const std::int64_t v = 3;
        f.add_attr("entities/4", "n", H5T_NATIVE_INT64, &v);
        f.check_refused("/entities/4 attribute 'n': is not a string");
    }
    SUBCASE("a rank-3 dataset in a group no entity owns")
    {
        fixture f;
        f.edit([](hid_t file) {
            const hid_t g  = H5Gcreate2(file, "extra", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const hid_t sp = simple_space({2, 2, 2});
            const hid_t ds = H5Dcreate2(g, "cube", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
            REQUIRE(ds >= 0);
            H5Dclose(ds);
            H5Sclose(sp);
            H5Gclose(g);
        });
        f.check_refused("/extra/cube: rank 3");
    }
    SUBCASE("a dangling soft link")
    {
        fixture f;
        f.edit([](hid_t file) {
            REQUIRE(H5Lcreate_soft("/nowhere", file, "entities/4/dangling", H5P_DEFAULT,
                                   H5P_DEFAULT) >= 0);
        });
        f.check_refused("/entities/4/dangling: a soft or external link");
    }
}

TEST_CASE("the walk only follows a tree of hard links, so a crafted file cannot loop or escape it")
{
    // A link back to an ancestor made the walk recurse until the stack
    // overflowed: `coast --load-state` on such a file died with SIGSEGV.
    SUBCASE("a soft link to the root")
    {
        fixture f;
        f.edit([](hid_t file) {
            REQUIRE(H5Lcreate_soft("/", file, "entities/loop", H5P_DEFAULT, H5P_DEFAULT) >= 0);
        });
        f.check_refused("/entities/loop: a soft or external link");
    }
    SUBCASE("a hard link to the root (a cycle)")
    {
        fixture f;
        f.edit([](hid_t file) {
            REQUIRE(H5Lcreate_hard(file, "/", file, "entities/loop", H5P_DEFAULT, H5P_DEFAULT) >= 0);
        });
        f.check_refused("/entities/loop: links to an object reached before");
    }
    SUBCASE("a group reached by two hard links (shared, not a cycle)")
    {
        fixture f;
        f.edit([](hid_t file) {
            REQUIRE(H5Lcreate_hard(file, "/entities/4", file, "entities/again", H5P_DEFAULT,
                                   H5P_DEFAULT) >= 0);
        });
        f.check_refused("links to an object reached before");
    }
    SUBCASE("an external link to another file")
    {
        fixture f;
        f.edit([&](hid_t file) {
            REQUIRE(H5Lcreate_external(f.path.string().c_str(), "/entities", file, "entities/ext",
                                       H5P_DEFAULT, H5P_DEFAULT) >= 0);
        });
        f.check_refused("/entities/ext: a soft or external link");
    }
    SUBCASE("groups nested deeper than the cap")
    {
        fixture f;
        f.edit([](hid_t file) {
            hid_t g = H5Gcreate2(file, "deep", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            for (int i = 0; i < 70; ++i) {
                const hid_t sub = H5Gcreate2(g, "d", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                REQUIRE(sub >= 0);
                H5Gclose(g);
                g = sub;
            }
            H5Gclose(g);
        });
        f.check_refused("nested deeper than 64 groups");
    }
}

TEST_CASE("a newer format version is reported as newer even when the file holds objects this build refuses")
{
    // The version gate runs before the whole-file check on both paths, so a
    // future format's new kinds of object do not mask the real reason.
    fixture f;
    f.edit([](hid_t file) {
        const hid_t md = H5Gopen2(file, "metadata", H5P_DEFAULT);
        REQUIRE(H5Adelete(md, "schema_version") >= 0);
        REQUIRE(hdf5::write_attr(md, "schema_version",
                                 std::to_string(logging::k_format_version + 1)) >= 0);
        H5Gclose(md);
    });
    const hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, 8);
    const hid_t sp = H5Screate(H5S_SCALAR);
    f.replace_dataset("/entities/4/s", st, sp, 8);
    H5Sclose(sp);
    H5Tclose(st);
    f.check_refused("newer than this build reads");
}

TEST_CASE("a refused runtime parse leaves no handle open and the file can be recreated")
{
    // parse_group held raw ids and closed them by hand, so a throw from deep in
    // the tree leaked one group per level plus the dataset — and kept the file
    // itself open (weak close degree), so H5Fcreate(TRUNC) on it failed.
    fixture f;
    const hid_t sp = simple_space({2, 2, 2});
    f.replace_dataset("/entities/4/m", H5T_NATIVE_FLOAT, sp, 8 * sizeof(float));
    H5Sclose(sp);
    REQUIRE_FALSE(state_logger::parse_state(f.path, f.h5).result.success);
    f.check_nothing_open();
    f.h5.submit_sync([&] {
        const hid_t file = H5Fcreate(f.path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(file >= 0);
        if (file >= 0) H5Fclose(file);
    });
}

TEST_CASE("mem_state_reader refuses a field of another rank or a group, like the HDF5 reader")
{
    logging::mem_group tree;
    tree.fields["s"] = 1.0;
    tree.fields["v"] = std::vector<float>{1.f, 2.f};
    tree.fields["m"] = logging::mem_array2d{{1.f, 2.f, 3.f, 4.f}, 2, 2};
    tree.groups["g"];
    logging::mem_state_reader r(tree, logging::k_format_version);
    logging::state_reader::array_2d_info info;

    CHECK(r.read_double("s", 0.0) == 1.0);
    CHECK(r.read_float_vec("v").size() == 2);
    CHECK(r.read_float_2d("m", info).size() == 4);
    CHECK(r.read_double("absent", 7.0) == 7.0);
    CHECK(r.read_float_vec("absent").empty());

    CHECK_THROWS_WITH_AS((void)r.read_double("v", 0.0),
                         doctest::Contains("/v: expected rank 0, the file has rank 1"), std::runtime_error);
    CHECK_THROWS_WITH_AS((void)r.read_bool("m", false),
                         doctest::Contains("/m: expected rank 0, the file has rank 2"), std::runtime_error);
    CHECK_THROWS_WITH_AS((void)r.read_float_vec("m"),
                         doctest::Contains("/m: expected rank 1, the file has rank 2"), std::runtime_error);
    CHECK_THROWS_WITH_AS((void)r.read_float_vec("s"),
                         doctest::Contains("/s: expected rank 1, the file has rank 0"), std::runtime_error);
    CHECK_THROWS_WITH_AS((void)r.read_float_2d("v", info),
                         doctest::Contains("/v: expected rank 2, the file has rank 1"), std::runtime_error);
    CHECK_THROWS_WITH_AS((void)r.read_float_vec("g"),
                         doctest::Contains("/g: is a group, not a dataset"), std::runtime_error);
}

TEST_CASE("the attribute helpers: absent is nullopt; malformed throws; well-formed reads")
{
    hdf5::executor h5;
    const fs::path path = fs::temp_directory_path() /
                          ("coast_attr_guards_" + std::to_string(::getpid()) + ".h5");
    h5.submit_sync([&] {
        auto f = hdf5::file::create(path.string());
        REQUIRE(f);
        const hid_t root = f.id();

        CHECK(hdf5::write_attr(root, "one", std::int64_t{42}) >= 0);
        CHECK(hdf5::write_attr(root, "str", std::string_view("abc")) >= 0);

        const hid_t sp = simple_space({3});
        const hid_t a  = H5Acreate2(root, "three", H5T_NATIVE_INT64, sp, H5P_DEFAULT, H5P_DEFAULT);
        const std::int64_t vals[3] = {1, 2, 3};
        REQUIRE(H5Awrite(a, H5T_NATIVE_INT64, vals) >= 0);
        H5Aclose(a);
        H5Sclose(sp);

        CHECK_FALSE(hdf5::read_attr_int64(root, "absent").has_value());
        CHECK_FALSE(hdf5::read_attr_string(root, "absent").has_value());
        CHECK(hdf5::read_attr_int64(root, "one") == 42);
        CHECK(hdf5::read_attr_double(root, "one") == 42.0);
        CHECK(hdf5::read_attr_string(root, "str") == "abc");

        CHECK_THROWS_WITH_AS((void)hdf5::read_attr_int64(root, "three"),
                             doctest::Contains("holds 3 elements, expected 1"), std::runtime_error);
        CHECK_THROWS_AS((void)hdf5::read_attr_uint64(root, "three"), std::runtime_error);
        CHECK_THROWS_AS((void)hdf5::read_attr_double(root, "three"), std::runtime_error);
        CHECK_THROWS_WITH_AS((void)hdf5::read_attr_string(root, "one"),
                             doctest::Contains("is not a string"), std::runtime_error);
        CHECK_THROWS_WITH_AS((void)hdf5::read_attr_int64(root, "str"),
                             doctest::Contains("read failed"), std::runtime_error);
    });
    fs::remove(path);
}

TEST_CASE("select_hyperslab refuses offset/count spans that do not match the rank")
{
    hdf5::executor h5;
    h5.submit_sync([] {
        const hsize_t dims[3] = {2, 2, 2};
        auto sp = hdf5::dataspace::simple(dims);
        const hsize_t off2[2] = {0, 0}, cnt2[2] = {1, 1};
        const hsize_t off3[3] = {0, 0, 0}, cnt3[3] = {1, 1, 1};
        CHECK(sp.select_hyperslab(off2, cnt2) < 0);
        CHECK(sp.select_hyperslab(off3, cnt3) >= 0);
    });
}
