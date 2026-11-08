// Micro-benchmark (NOT a ctest) — isolates observe_writer_t throughput from
// any simulation cost. Drives the real writer with a wide (array-shaped) site
// for many rows and times start()..stop_and_join() (which forces the full
// executor drain + flush). This is where the chunk-cache thrash shows: with
// the historical 1024-row chunk a wide row produces a chunk far larger than
// the 1 MiB cache, so every single-row append triggers a read-modify-write of
// the whole chunk. The byte-budget chunk geometry keeps the chunk cache-
// resident.
//
// Usage: bench_observe_writer [width_floats] [rows]
// Prints: rows, width, MiB written, seconds, MiB/s, rows/s.

#include "observe/observe_writer.h"
#include "observe/observe_site.h"
#include "hdf5/executor.h"

#include <hdf5.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <vector>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const std::size_t   width = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 4096;
    const std::uint64_t rows  = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 4000;

    const fs::path h5_path = fs::temp_directory_path() / "coast_bench_writer.h5";
    std::error_code ec;
    fs::remove(h5_path, ec);

    observe::observe_site_t site;
    site.full_id     = "bench/wide";
    site.h5_dtype    = H5T_NATIVE_FLOAT;
    site.shape       = {static_cast<hsize_t>(width)};
    site.value_bytes = width * sizeof(float);

    const std::vector<float> row(width, 1.0f);

    const auto t0 = std::chrono::steady_clock::now();
    {
        hdf5::executor exec;
        observe::observe_writer_t writer(h5_path, exec);
        writer.start();
        // enqueue() takes the value + its byte size and copies both into an
        // owned buffer on the executor thread.
        for (std::uint64_t s = 0; s < rows; ++s) {
            writer.enqueue(&site, row.data(), site.value_bytes, s);
        }
        writer.stop_and_join();   // forces full drain + flush + close
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double mib  = static_cast<double>(rows) * site.value_bytes / (1024.0 * 1024.0);
    std::printf("rows=%llu width=%zu  %.1f MiB  %.3f s  %.1f MiB/s  %.0f rows/s\n",
                static_cast<unsigned long long>(rows), width, mib, secs,
                mib / secs, static_cast<double>(rows) / secs);

    fs::remove(h5_path, ec);
    return 0;
}
