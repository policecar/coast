// h5browse chunk_cache — LRU order, byte budget, shared_ptr liveness.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/chunk_cache.h"

#include <cstdint>
#include <memory>

using h5b::chunk_cache;
using h5b::row_block;

namespace {

std::shared_ptr<row_block> make_block(std::uint32_t dataset,
                                      std::uint64_t block, std::size_t bytes)
{
    auto b       = std::make_shared<row_block>();
    b->dataset   = dataset;
    b->block     = block;
    b->n_rows    = bytes;
    b->row_width = 1;
    b->bytes.resize(bytes, std::byte{static_cast<unsigned char>(block & 0xff)});
    return b;
}

} // namespace

TEST_CASE("chunk_cache: hit, miss, and replacement")
{
    chunk_cache cache(1 << 20);
    CHECK(cache.get(0, 0) == nullptr);

    cache.put(make_block(0, 0, 100));
    const auto hit = cache.get(0, 0);
    REQUIRE(hit != nullptr);
    CHECK(hit->bytes.size() == 100);
    CHECK(cache.blocks_cached() == 1);

    // Same key replaces, byte accounting follows.
    cache.put(make_block(0, 0, 300));
    CHECK(cache.blocks_cached() == 1);
    CHECK(cache.bytes_cached() == 300);

    // Distinct datasets with the same block index do not collide.
    cache.put(make_block(1, 0, 50));
    CHECK(cache.get(0, 0)->bytes.size() == 300);
    CHECK(cache.get(1, 0)->bytes.size() == 50);
}

TEST_CASE("chunk_cache: LRU eviction under byte budget, recency via get")
{
    chunk_cache cache(1000);
    cache.put(make_block(0, 1, 400));
    cache.put(make_block(0, 2, 400));

    // Touch block 1 -> block 2 becomes LRU.
    CHECK(cache.get(0, 1) != nullptr);

    cache.put(make_block(0, 3, 400)); // 1200 > 1000: evict LRU (block 2)
    CHECK(cache.get(0, 2) == nullptr);
    CHECK(cache.get(0, 1) != nullptr);
    CHECK(cache.get(0, 3) != nullptr);
    CHECK(cache.bytes_cached() == 800);
}

TEST_CASE("chunk_cache: oversized block is kept and evicts everything else")
{
    chunk_cache cache(1000);
    cache.put(make_block(0, 1, 400));
    cache.put(make_block(0, 2, 4000)); // alone above budget

    CHECK(cache.get(0, 1) == nullptr);
    const auto big = cache.get(0, 2);
    REQUIRE(big != nullptr);
    CHECK(big->bytes.size() == 4000);
    CHECK(cache.blocks_cached() == 1);
}

TEST_CASE("chunk_cache: consumer's shared_ptr survives eviction and clear")
{
    chunk_cache cache(1000);
    cache.put(make_block(7, 42, 600));
    const std::shared_ptr<const row_block> held = cache.get(7, 42);
    REQUIRE(held != nullptr);

    cache.put(make_block(7, 43, 600)); // evicts 42
    CHECK(cache.get(7, 42) == nullptr);
    CHECK(held->bytes.size() == 600); // still alive for the frame drawing it
    CHECK(held->bytes[0] == std::byte{42});

    cache.clear();
    CHECK(cache.bytes_cached() == 0);
    CHECK(held->bytes.size() == 600);
}
