#include "h5browse/chunk_cache.h"

#include <cassert>

namespace h5b {

chunk_cache::chunk_cache(std::size_t byte_budget) : budget_(byte_budget) {}

chunk_cache::key_t chunk_cache::key_of(std::uint32_t dataset,
                                       std::uint64_t block)
{
    // 2^24 datasets x 2^40 blocks; observe files top out around 2^11
    // datasets and multi-TB files around 2^30 blocks — ample margin.
    assert(dataset < (1u << 24));
    assert(block < (1ull << 40));
    return (static_cast<std::uint64_t>(dataset) << 40) | block;
}

std::shared_ptr<const row_block> chunk_cache::get(std::uint32_t dataset,
                                                  std::uint64_t block)
{
    const key_t                 k = key_of(dataset, block);
    std::scoped_lock lock(mutex_);
    const auto       it = map_.find(k);
    if (it == map_.end()) return nullptr;
    lru_.splice(lru_.begin(), lru_, it->second); // touch: move to MRU
    return it->second->second.blockp;
}

void chunk_cache::put(std::shared_ptr<const row_block> b)
{
    assert(b != nullptr);
    const key_t      k = key_of(b->dataset, b->block);
    std::scoped_lock lock(mutex_);

    if (const auto it = map_.find(k); it != map_.end()) {
        bytes_ -= it->second->second.blockp->bytes.size();
        lru_.erase(it->second);
        map_.erase(it);
    }
    bytes_ += b->bytes.size();
    lru_.emplace_front(k, entry{std::move(b)});
    map_[k] = lru_.begin();
    evict_to_budget_locked();
}

void chunk_cache::evict_to_budget_locked()
{
    // Never evict the MRU entry — a block above the whole budget must
    // still be usable by the consumer that requested it.
    while (bytes_ > budget_ && lru_.size() > 1) {
        auto& [k, e] = lru_.back();
        bytes_ -= e.blockp->bytes.size();
        map_.erase(k);
        lru_.pop_back();
    }
}

void chunk_cache::clear()
{
    std::scoped_lock lock(mutex_);
    lru_.clear();
    map_.clear();
    bytes_ = 0;
}

std::size_t chunk_cache::bytes_cached() const
{
    std::scoped_lock lock(mutex_);
    return bytes_;
}

std::size_t chunk_cache::blocks_cached() const
{
    std::scoped_lock lock(mutex_);
    return lru_.size();
}

} // namespace h5b
