# gui_replay unit tests — the --log-buffers reader behind the replay dialog.

# Shape checks against crafted buffer logs (#150).
add_executable(test_buffer_log_reader test_buffer_log_reader.cpp)
target_link_libraries(test_buffer_log_reader PRIVATE doctest::doctest coast_gui)
add_test(NAME unit.gui_replay.buffer_log_reader COMMAND test_buffer_log_reader)
