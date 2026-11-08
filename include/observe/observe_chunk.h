//
// Created for the COAST observe system
//
// Chunk-geometry policy for the observe HDF5 writer.
//
// The value datasets are extensible along the time axis and appended one row
// per observed step. HDF5 caches whole chunks: as long as a chunk fits inside
// the dataset's chunk cache (rdcc_nbytes) the single-row appends accumulate in
// memory and flush lazily. But the moment one chunk exceeds the cache, HDF5
// stops caching it and writes each single-row append directly through to the
// file, losing the batching a cache-resident chunk would give.
//
// This is NOT a full-chunk read-modify-write per row, so it is NOT O(rows^2):
// measured in an isolated writer micro-benchmark, it is a bounded
// constant-factor throughput loss that grows with row width — ~1.9x at a
// 16 KiB row, ~2.6x at 64 KiB (HDF5 does more, smaller direct writes when it
// bypasses the cache). An earlier hypothesis framed this as ~O(rows^2) thrash;
// the benchmark walked that back to the constant factor above.
//
// The historical writer hardcoded the time-axis chunk to 1024 rows regardless
// of per-row size, so any site whose row exceeded ~1 KiB (e.g. an array-shaped
// population output) produced chunks far larger than the 1 MiB default cache
// and thrashed. pick_chunk_rows() sizes the time-axis chunk from a byte budget
// instead, keeping each chunk near `target_chunk_bytes` so it stays cache-
// resident. Scalar / small-row sites are clamped to `max_rows` (the historical
// 1024), so their on-disk geometry is unchanged.
//
// Pure integer arithmetic, no HDF5 dependency, so it is unit-testable in
// isolation.
//
#ifndef COAST_OBSERVE_CHUNK_H
#define COAST_OBSERVE_CHUNK_H

#include <cstddef>
#include <cstdint>

namespace observe {

/// Number of rows along the extensible time axis to put in one chunk.
///
/// `row_bytes`  = bytes of one observed value (sizeof(scalar) * product(shape)).
/// `target_chunk_bytes` = desired chunk size in bytes (default 1 MiB, matching
///                        HDF5's default chunk cache so one chunk fits).
/// `max_rows`   = upper clamp on rows-per-chunk (default 1024 = historical
///                geometry, preserved for scalar/small sites).
///
/// Returns a value in [1, max_rows]. A zero `row_bytes` (degenerate site)
/// returns `max_rows` and avoids a divide-by-zero.
inline std::uint64_t pick_chunk_rows(std::size_t   row_bytes,
                                     std::size_t   target_chunk_bytes = (std::size_t{1} << 20),
                                     std::uint64_t max_rows           = 1024) {
    if (row_bytes == 0) {
        return max_rows;
    }
    std::uint64_t rows = static_cast<std::uint64_t>(target_chunk_bytes / row_bytes);
    if (rows < 1) {
        rows = 1;
    }
    if (rows > max_rows) {
        rows = max_rows;
    }
    return rows;
}

} // namespace observe

#endif // COAST_OBSERVE_CHUNK_H
