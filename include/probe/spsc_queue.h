//
// Created for the COAST probe system
//

#ifndef COAST_SPSC_QUEUE_H
#define COAST_SPSC_QUEUE_H

#include <vector>
#include <atomic>
#include <cstddef>

namespace probe {

/// Lock-free single-producer single-consumer queue.
/// Used for passing probe values from simulation thread to GUI thread.
template<typename T>
class spsc_queue {
    std::vector<T> buffer_;
    std::atomic<std::size_t> head_{0};  // Written by producer
    std::atomic<std::size_t> tail_{0};  // Written by consumer
    std::size_t capacity_;

public:
    explicit spsc_queue(std::size_t capacity = 4096)
        : buffer_(capacity), capacity_(capacity) {}

    /// Producer: try to push a value. Returns false if queue is full.
    bool try_push(T value) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t next = (head + 1) % capacity_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Full - drop value
        }
        buffer_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// Consumer: try to pop a value. Returns false if queue is empty.
    bool try_pop(T& out) {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Empty
        }
        out = buffer_[tail];
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return true;
    }

    /// Consumer: drain all available values to a callback.
    template<typename Fn>
    void drain(Fn&& fn) {
        T value;
        while (try_pop(value)) {
            fn(value);
        }
    }

    /// Clear the queue (consumer side).
    void clear() {
        tail_.store(head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    /// Check if empty (approximate, for diagnostics only).
    [[nodiscard]] bool empty() const {
        return tail_.load(std::memory_order_relaxed) ==
               head_.load(std::memory_order_relaxed);
    }

    /// Get approximate size (for diagnostics only).
    [[nodiscard]] std::size_t size_approx() const {
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (head >= tail) {
            return head - tail;
        }
        return capacity_ - tail + head;
    }
};

} // namespace probe

#endif // COAST_SPSC_QUEUE_H
