add_executable(test_step_error test_step_error.cpp)
target_link_libraries(test_step_error PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.core.step_error COMMAND test_step_error)

# demand_published: the single-slot DEMAND-DRIVEN handoff primitive — the
# producer recomputes only after the consumer has taken the last value. (Not to
# be confused with h5browse's latest_value mailbox, which publishes
# unconditionally; scan_engine.h documents why it needs that one instead.)
#
# It lives in core/ because it is GUI-free — plain atomics and a mutex, no ImGui
# type anywhere in it — not because the core uses it: today its only consumers
# are gui_core's buffer_browser and gui_manager. Keeping the primitive and this
# test in core means the concurrency contract is exercised in every build
# configuration, including headless, where the GUI consumers do not exist.
add_executable(test_demand_published test_demand_published.cpp)
target_link_libraries(test_demand_published PRIVATE doctest::doctest coast_lib coast_test_support)
add_test(NAME unit.core.demand_published COMMAND test_demand_published
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# seed_t / seed_offset_t: the 32-bit seed rule (composition modulo 2^32, the
# [-2^31, 2^32 - 1] input range) and its nlohmann hooks.
add_executable(test_seed test_seed.cpp)
target_link_libraries(test_seed PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.core.seed COMMAND test_seed)
