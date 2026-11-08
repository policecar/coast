#ifndef COAST_CHUNK_CACHE_H
#define COAST_CHUNK_CACHE_H

// Chunk-aligned LRU row-block cache for the table view and pairwise
// gather (IDD D5). Byte-budgeted (512 MB default); consumers receive
// shared_ptr<const row_block>, so a block evicted mid-frame survives the
// frame drawing it. One mutex inside — accessed from the GUI thread
// (get, via scan_engine::try_get_block) and the scan thread (put/get),
// never from executor jobs.

#include "h5browse/scalar_type.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace h5b {

struct row_block {
    std::uint32_t dataset = 0; // index into file_model::datasets
    std::uint64_t block   = 0; // block index within the dataset

    std::uint64_t first_row = 0;
    std::uint64_t n_rows    = 0;
    std::uint64_t row_width = 1;
    scalar_type   dtype     = scalar_type::f64;

    std::vector<std::byte> bytes; // native dtype, row-major (D10)
};

class chunk_cache {
public:
    static constexpr std::size_t default_budget = 512ull << 20; // 512 MB

    explicit chunk_cache(std::size_t byte_budget = default_budget);

    // nullptr on miss; a hit refreshes LRU recency.
    [[nodiscard]] std::shared_ptr<const row_block> get(std::uint32_t dataset,
                                                       std::uint64_t block);

    // Inserts (or replaces) and evicts least-recently-used blocks until the
    // budget holds. A single block larger than the whole budget is still
    // stored (the cache would otherwise be useless for that dataset); it
    // then evicts everything else.
    void put(std::shared_ptr<const row_block> b);

    void clear();

    [[nodiscard]] std::size_t bytes_cached() const;
    [[nodiscard]] std::size_t blocks_cached() const;

private:
    using key_t = std::uint64_t; // dataset << 40 | block

    static key_t key_of(std::uint32_t dataset, std::uint64_t block);

    void evict_to_budget_locked();

    mutable std::mutex mutex_;
    std::size_t        budget_;
    std::size_t        bytes_ = 0;

    // MRU at front. map values hold list iterators for O(1) touch.
    struct entry {
        std::shared_ptr<const row_block> blockp;
    };
    std::list<std::pair<key_t, entry>> lru_;
    std::unordered_map<key_t, std::list<std::pair<key_t, entry>>::iterator> map_;
};

} // namespace h5b

#endif // COAST_CHUNK_CACHE_H
