# Defines, per integration test registered in COAST_INTEGRATION_TESTS:
#   coast_update_goldens_<NAME>  — capture + rotate for that one test
# Plus a meta target:
#   coast_update_goldens          — depends on all per-test targets
#
# Backed by scripts/update_golden.sh.

get_property(_its GLOBAL PROPERTY COAST_INTEGRATION_TESTS)
set(_all_update_targets)
foreach(_entry IN LISTS _its)
    string(REPLACE "|" ";" _parts "${_entry}")
    list(GET _parts 0 _name)
    list(GET _parts 1 _exe)
    list(GET _parts 2 _blob)
    list(GET _parts 3 _goldens)

    add_custom_target(coast_update_goldens_${_name}
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/update_golden.sh
                ${_exe} ${_blob} ${_goldens}
        DEPENDS integration_${_name}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
        VERBATIM
        COMMENT "Capturing golden for ${_name}"
    )
    list(APPEND _all_update_targets coast_update_goldens_${_name})
endforeach()

if(_all_update_targets)
    add_custom_target(coast_update_goldens DEPENDS ${_all_update_targets})
endif()
