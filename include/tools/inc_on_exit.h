//
// Created by claude on 29.01.26.
//

#ifndef INC_ON_EXIT_H
#define INC_ON_EXIT_H

#include <cstddef>

namespace coast {

/**
 * RAII wrapper that increments a counter on destruction.
 * Useful for ensuring a counter is incremented at function exit
 * regardless of which return path is taken.
 */
class inc_on_exit {
public:
    explicit inc_on_exit(std::size_t& counter) : counter_(counter) {}
    ~inc_on_exit() { ++counter_; }

    // Non-copyable, non-movable
    inc_on_exit(const inc_on_exit&) = delete;
    inc_on_exit& operator=(const inc_on_exit&) = delete;
    inc_on_exit(inc_on_exit&&) = delete;
    inc_on_exit& operator=(inc_on_exit&&) = delete;

private:
    std::size_t& counter_;
};

} // namespace coast

#endif //INC_ON_EXIT_H
