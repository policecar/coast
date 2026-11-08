#ifndef COAST_DEMAND_PUBLISHED_H
#define COAST_DEMAND_PUBLISHED_H

#include <atomic>
#include <mutex>
#include <utility>

namespace coast {

/**
 * Single-slot, demand-driven sim -> GUI publisher.
 *
 * The producer (sim thread, inside a post_swap hook) calls produce_if_wanted():
 * it recomputes and republishes ONLY when the consumer has taken the previous
 * value. So the (potentially expensive) compute runs at the consumer's
 * consumption rate (~GUI frame rate), never the much higher sim step rate. When
 * the consumer is idle (window closed, GUI stalled) the producer early-returns on
 * a single atomic load — no wasted work.
 *
 * The consumer (GUI thread) calls consume() once per frame to read the latest
 * value and request the next one.
 *
 * Threading contract: exactly one producer thread and one consumer thread. Two
 * independent mechanisms cooperate — `fresh_` (atomic) decides WHETHER to
 * produce; the mutex guards `value_` against a torn read. The flag never
 * protects the data and the mutex never gates production, so neither can starve
 * the other.
 *
 * Lifetime: hold this via shared_ptr and capture that shared_ptr (by value) in
 * the producer hook, so the slot's lifetime does not depend on when the owner
 * removes the hook (removal is synchronous since #235).
 */
template <class T>
class demand_published {
public:
    /// Producer (sim thread). Recompute + publish only if the consumer has taken
    /// the last value; otherwise return immediately. `compute` runs with no lock
    /// held, so a slow compute never blocks consume().
    template <class F>
    void produce_if_wanted(F&& compute)
    {
        if (fresh_.load(std::memory_order_acquire)) {
            return;   // last value not consumed yet — skip the expensive work
        }
        T v = std::forward<F>(compute)();
        {
            std::lock_guard lock(mutex_);
            value_ = std::move(v);
        }
        fresh_.store(true, std::memory_order_release);
    }

    /// Consumer (GUI thread). Read the latest published value and request the
    /// next. Returns the most recent value even if the producer has not run since
    /// the last consume (the slot always holds the latest publish).
    [[nodiscard]] T consume()
    {
        T out;
        {
            std::lock_guard lock(mutex_);
            out = value_;
        }
        fresh_.store(false, std::memory_order_release);
        return out;
    }

private:
    std::mutex        mutex_;
    T                 value_{};
    std::atomic<bool> fresh_{false};   // false => consumer wants a fresh value
};

} // namespace coast

#endif // COAST_DEMAND_PUBLISHED_H
