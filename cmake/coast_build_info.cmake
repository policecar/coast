# Capture git SHA + dirty-state at CMake configure time.
# Re-runs on every cmake configure (not every build) — adequate for run-manifest
# provenance. To force re-capture, run `cmake --fresh -B build ...` or touch
# CMakeLists.txt.

find_package(Git QUIET)
if (Git_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE COAST_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_sha_rc
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE _git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_status_rc
    )
    if (NOT _git_sha_rc EQUAL 0)
        set(COAST_GIT_SHA "unknown")
    endif()
    if (_git_status_rc EQUAL 0 AND "${_git_status}" STREQUAL "")
        set(COAST_GIT_DIRTY_BOOL "false")
    else()
        set(COAST_GIT_DIRTY_BOOL "true")
    endif()
else()
    set(COAST_GIT_SHA          "unknown")
    set(COAST_GIT_DIRTY_BOOL   "true")
endif()

# Ensure CMAKE_BUILD_TYPE has a default (Release per project CLAUDE.md).
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# The toolchain, for comparing results across hosts: bit-identical output across
# machines was measured to need the same compiler AND the same target CPU (the
# whole project builds with -march=native, CMakeLists.txt), so a coast_runtime.json
# records both. COAST_TARGET_CPU is what -march=native resolves to on the
# configuring host, asked from the compiler itself; "unknown" if it will not say.
set(COAST_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(COAST_TARGET_CPU "unknown")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # x86 prints the resolved CPU under -march= (and a blank -mcpu=); aarch64 GCC
    # leaves -march= blank and prints it under -mcpu=. A blank value never matches,
    # and options are listed alphabetically, so a filled -march= wins over -mcpu=.
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -march=native -Q --help=target
                    OUTPUT_VARIABLE _coast_target ERROR_QUIET RESULT_VARIABLE _coast_rc)
    if (_coast_rc EQUAL 0 AND _coast_target MATCHES "[\r\n][ \t]*-m(arch|cpu)=[ \t]+([^ \t\r\n]+)")
        set(COAST_TARGET_CPU "${CMAKE_MATCH_2}")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")   # Clang and AppleClang
    # "-###" must stay quoted: unquoted, CMake reads the # as a comment and drops
    # the rest of the line, so clang ran without an input and the match never fired.
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -march=native "-###" -x c++ -c /dev/null
                    ERROR_VARIABLE _coast_target OUTPUT_QUIET RESULT_VARIABLE _coast_rc)
    if (_coast_rc EQUAL 0 AND _coast_target MATCHES [=["-target-cpu" "([^"]+)"]=])
        set(COAST_TARGET_CPU "${CMAKE_MATCH_1}")
    endif()
endif()
if (COAST_TARGET_CPU STREQUAL "unknown")
    message(WARNING "could not ask ${COAST_COMPILER} what -march=native resolves to; "
                    "coast_runtime.json will record coast_target_cpu = \"unknown\"")
endif()

# Echo what we captured at configure time — improves audit/repro logs when a
# downstream user needs to know which COAST commit a coast_runtime.json was
# emitted from.
message(STATUS "COAST build info captured: git_sha=${COAST_GIT_SHA} dirty=${COAST_GIT_DIRTY_BOOL} build_type=${CMAKE_BUILD_TYPE} compiler=${COAST_COMPILER} target_cpu=${COAST_TARGET_CPU}")

configure_file(
    ${CMAKE_SOURCE_DIR}/include/sim_core/coast_build_info.h.in
    ${CMAKE_BINARY_DIR}/generated/sim_core/coast_build_info.h
    @ONLY
)
