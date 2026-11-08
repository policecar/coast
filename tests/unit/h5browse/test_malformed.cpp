// h5browse G6 zero-crash contract: malformed inputs are refused with a
// clean, triage-grade h5b::error — never a crash, abort, or hang.
//
// This is the fast in-repo subset of the T3 corpus taxonomy
// (.dew/design-verification, 55/55 clean on libhdf5 2.1.1); the full
// 46-case seeded corpus remains a manual regression asset. Corruptions
// here are derived from a REAL observe file written in-process.
//
// The "truncated"/"being written" substring checks pin the triage mapping
// in h5b_error.cpp to libhdf5's message vocabulary — if a library upgrade
// rewords its stack, these tests are the intended tripwire.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/file_model.h"
#include "hdf5/executor.h"
#include "observe/observe_site.h"
#include "observe/observe_writer.h"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path temp_path(const std::string& name)
{
    const fs::path  p = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove(p, ec);
    return p;
}

// H5T_NATIVE_* macros call H5open() on the CALLING thread; route through
// the executor so only the worker ever initializes libhdf5 (the H5VL init
// assertion fires if two threads race H5open()).
hid_t native_float(hdf5::executor& exec)
{
    return exec.submit_sync([] { return H5T_NATIVE_FLOAT; });
}

// One real observe file, written once per process by the real writer.
const fs::path& pristine_file(hdf5::executor& exec)
{
    static fs::path path = [&] {
        const fs::path p = temp_path("h5b_mal_pristine.h5");
        float          scratch = 0.0f;
        observe::observe_site_t site;
        site.full_id     = "neuron[0]/activity";
        site.h5_dtype    = native_float(exec);
        site.shape       = {};
        site.value_bytes = sizeof(float);

        observe::observe_writer_t writer(p, exec);
        writer.start();
        for (std::uint64_t s = 0; s < 100; ++s) {
            scratch = static_cast<float>(s) * 0.25f;
            writer.enqueue(&site, &scratch, sizeof(scratch), s);
        }
        writer.stop_and_join();
        return p;
    }();
    return path;
}

std::vector<char> read_all(const fs::path& p)
{
    std::ifstream     in(p, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    return bytes;
}

void write_all(const fs::path& p, const std::vector<char>& bytes)
{
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Every malformed case funnels through here: must fail by VALUE with a
// non-empty message. Reaching the end of the function IS the zero-crash
// assertion.
h5b::error expect_clean_rejection(hdf5::executor& exec, const fs::path& p)
{
    const auto r = h5b::open_and_map(exec, p);
    REQUIRE_FALSE(r.has_value());
    CHECK_FALSE(r.error().message.empty());
    return r.error();
}

} // namespace

TEST_CASE("malformed: pristine control opens (corpus baseline is valid)")
{
    hdf5::executor exec;
    const auto     r = h5b::open_and_map(exec, pristine_file(exec));
    REQUIRE(r.has_value());
    CHECK(r->datasets.size() == 1);
    CHECK(r->datasets[0].rows == 100);
}

TEST_CASE("malformed: truncation ladder is refused with truncation triage text")
{
    hdf5::executor exec;
    const auto     whole = read_all(pristine_file(exec));
    REQUIRE(whole.size() > 2048);

    // 512 B (header fragment), half the file, all-but-one-byte.
    const std::size_t sizes[] = {512, whole.size() / 2, whole.size() - 1};
    for (const std::size_t sz : sizes) {
        const fs::path p = temp_path("h5b_mal_trunc_" + std::to_string(sz) + ".h5");
        write_all(p, {whole.begin(), whole.begin() + static_cast<long>(sz)});

        const h5b::error e = expect_clean_rejection(exec, p);
        CHECK(e.category == h5b::error_category::open);
        CHECK(e.message.find("truncated") != std::string::npos);

        std::error_code ec;
        fs::remove(p, ec);
    }
}

TEST_CASE("malformed: zero-length, text, and signature+garbage files")
{
    hdf5::executor exec;

    const fs::path empty = temp_path("h5b_mal_empty.h5");
    write_all(empty, {});
    expect_clean_rejection(exec, empty);

    const fs::path text = temp_path("h5b_mal_text.h5");
    const std::string msg = "this is not an HDF5 file at all\n";
    write_all(text, {msg.begin(), msg.end()});
    expect_clean_rejection(exec, text);

    // Valid 8-byte HDF5 signature followed by garbage.
    const fs::path    sig = temp_path("h5b_mal_sig.h5");
    std::vector<char> bytes{'\x89', 'H', 'D', 'F', '\r', '\n', '\x1a', '\n'};
    for (int i = 0; i < 4096; ++i) {
        bytes.push_back(static_cast<char>((i * 37) & 0xff));
    }
    write_all(sig, bytes);
    expect_clean_rejection(exec, sig);

    std::error_code ec;
    fs::remove(empty, ec);
    fs::remove(text, ec);
    fs::remove(sig, ec);
}

TEST_CASE("malformed: superblock bit flip is refused (metadata checksums)")
{
    hdf5::executor exec;
    auto           bytes = read_all(pristine_file(exec));
    REQUIRE(bytes.size() > 64);
    bytes[20] = static_cast<char>(bytes[20] ^ 0x40); // inside the superblock

    const fs::path p = temp_path("h5b_mal_bitflip.h5");
    write_all(p, bytes);
    expect_clean_rejection(exec, p);

    std::error_code ec;
    fs::remove(p, ec);
}

TEST_CASE("live writer, SAME process: open succeeds on the shared file object")
{
    // The T3 "refused by file lock" finding is CROSS-process only (flock is
    // advisory between processes). Within one process libhdf5 recognizes
    // the already-open file and shares the coherent in-memory file object,
    // so a RDONLY open of a file our own sim is writing SUCCEEDS — the
    // browser sees a consistent snapshot as of its open. This test pins
    // that in-process semantic; cross-process refusal (and the lock triage
    // text in h5b_error.cpp) is evidenced by the T3 corpus run.
    hdf5::executor exec;
    const fs::path p = temp_path("h5b_mal_live.h5");

    float                   scratch = 0.0f;
    observe::observe_site_t site;
    site.full_id     = "x";
    site.h5_dtype    = native_float(exec);
    site.shape       = {};
    site.value_bytes = sizeof(float);

    observe::observe_writer_t writer(p, exec);
    writer.start();
    writer.enqueue(&site, &scratch, sizeof(scratch), 0);
    writer.enqueue(&site, &scratch, sizeof(scratch), 1);

    // Writer still holds the file open for write; FIFO order guarantees
    // both appends landed before the open job runs.
    const auto live = h5b::open_and_map(exec, p);
    REQUIRE(live.has_value());
    REQUIRE(live->datasets.size() == 1);
    CHECK(live->datasets[0].rows == 2); // snapshot as of the open

    writer.enqueue(&site, &scratch, sizeof(scratch), 2); // appends continue after our open

    writer.stop_and_join();

    // After the writer closed, a fresh open sees the full extent.
    const auto closed = h5b::open_and_map(exec, p);
    REQUIRE(closed.has_value());
    CHECK(closed->datasets[0].rows == 3);

    std::error_code ec;
    fs::remove(p, ec);
}
