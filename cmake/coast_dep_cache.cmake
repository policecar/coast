# A machine-wide cache of downloaded dependencies, tried before the network.
#
# Every new build tree used to download its dependencies again: a new branch in CI,
# the same branch on another runner, every coastd worktree. Each of those
# downloads depended on the remote host being up; on 2026-11-08 GitHub's release
# downloads answered 503 for minutes and failed CI configures that way.
#
# coast_cached_download(<out-var> <url> <sha256> <subdir>) returns a path to a
# verified copy in ${COAST_DEPS_CACHE}/<subdir>/, downloading it there once if it
# is missing. Only the downloaded file is shared: each build tree still extracts
# it into its own _deps through its own FetchContent subbuild, so trees with
# different generators or CMake versions do not collide (the reason a shared
# FetchContent dir was rejected). The copy is checked against <sha256> on every
# configure; the caller passes the same hash to FetchContent, so a wrong file is
# refused there too.
#
# If the cache is disabled (COAST_DEPS_CACHE empty), not writable, or the download
# fails, <out-var> is <url> and FetchContent downloads it as before.

if(DEFINED ENV{COAST_DEPS_CACHE})
    set(_coast_deps_default "$ENV{COAST_DEPS_CACHE}")
elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_coast_deps_default "$ENV{HOME}/.cache/coast/deps")
else()
    set(_coast_deps_default "")
endif()
set(COAST_DEPS_CACHE "${_coast_deps_default}" CACHE PATH
    "Machine-wide cache of downloaded dependencies, tried before the network (empty: off)")

function(coast_cached_download out url sha256 subdir)
    set(${out} "${url}" PARENT_SCOPE)
    if(NOT COAST_DEPS_CACHE)
        return()
    endif()
    get_filename_component(name "${url}" NAME)
    set(dir  "${COAST_DEPS_CACHE}/${subdir}")
    set(file "${dir}/${name}")

    if(EXISTS "${file}")
        file(SHA256 "${file}" have)
        if(have STREQUAL sha256)
            message(STATUS "Dependency cache: ${file}")
            set(${out} "${file}" PARENT_SCOPE)
            return()
        endif()
        message(WARNING "Dependency cache: ${file} has SHA256 ${have}, expected ${sha256}; "
                        "downloading it again")
        file(REMOVE "${file}")
    endif()

    # Not file(MAKE_DIRECTORY): it is fatal on failure, and an unwritable cache
    # must only fall back to the network.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${dir}"
                    RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
    if(NOT rc EQUAL 0)
        message(STATUS "Dependency cache: cannot create ${dir}; downloading ${url} directly")
        return()
    endif()

    # Download under a unique name, then rename: two configures filling the cache
    # at once each write their own file, and the rename is atomic.
    string(RANDOM LENGTH 12 tag)
    set(part "${file}.${tag}.part")
    # INACTIVITY_TIMEOUT: a frontend that accepts the connection and then sends
    # nothing (GitHub's fra edge on 2026-11-08) would otherwise hang the
    # configure. It ends only a stalled transfer, not a slow one.
    message(STATUS "Dependency cache: downloading ${url}")
    file(DOWNLOAD "${url}" "${part}" STATUS st TLS_VERIFY ON INACTIVITY_TIMEOUT 60)
    list(GET st 0 code)
    if(NOT code EQUAL 0)
        list(GET st 1 msg)
        file(REMOVE "${part}")
        message(STATUS "Dependency cache: download failed (${msg}); trying ${url} directly")
        return()
    endif()
    file(SHA256 "${part}" have)
    if(NOT have STREQUAL sha256)
        file(REMOVE "${part}")
        message(STATUS "Dependency cache: downloaded SHA256 ${have}, expected ${sha256}; "
                       "trying ${url} directly")
        return()
    endif()
    file(RENAME "${part}" "${file}")
    set(${out} "${file}" PARENT_SCOPE)
endfunction()
