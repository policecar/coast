# console unit tests — embedded clang-repl interpreter over the console_api
# facade. Only built when COAST_WITH_CONSOLE is ON (needs system LLVM/Clang).
#
# -rdynamic is REQUIRED: the JIT resolves the compiled facade symbols against
# the host process's dynamic symbol table.

# Needs the daemon EXECUTABLE, so it is guarded like the coastd real-daemon tests:
# $<TARGET_FILE:coastd> on a missing target is a configure error, not a skip.
if(TARGET coastd)
    add_executable(test_console_repl test_console_repl.cpp)
    target_link_libraries(test_console_repl PRIVATE doctest::doctest coast_console)
    target_link_options(test_console_repl PRIVATE -rdynamic)
    # test_daemon.h uses the load-independent wait helpers in tests/support.
    target_include_directories(test_console_repl PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
    # Drives the coastd console façade against a live daemon over the JIT.
    target_compile_definitions(test_console_repl PRIVATE COASTD_BIN="$<TARGET_FILE:coastd>")
    add_test(NAME unit.console.repl COMMAND test_console_repl)
    # COASTD_BIN is a PATH, not a build edge: a generator expression in a compile
    # definition creates NO dependency, so without this the test happily runs against
    # a stale coastd from a previous build and "proves" whatever that binary does.
    # tests/unit/coastd/CMakeLists.txt documents the same trap three times; this
    # directory was missed.
    add_dependencies(test_console_repl coastd)
endif()

add_executable(test_console_session test_console_session.cpp)
target_link_libraries(test_console_session PRIVATE doctest::doctest coast_console)
target_link_options(test_console_session PRIVATE -rdynamic)
add_test(NAME unit.console.session COMMAND test_console_session)

# quote_cpp is a header-only string helper — no interpreter, no JIT, so this test
# links only doctest and needs neither coast_console nor -rdynamic.
add_executable(test_quote_cpp test_quote_cpp.cpp)
target_link_libraries(test_quote_cpp PRIVATE doctest::doctest)
target_include_directories(test_quote_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME unit.console.quote_cpp COMMAND test_quote_cpp)
