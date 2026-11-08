# scripts/update_golden.sh: per-golden sidecars, rotation, refusals. Runs the
# script against a fake capture executable and checks with the real coast_compare.
add_test(NAME unit.scripts.update_golden
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_update_golden.sh
                 ${CMAKE_SOURCE_DIR}/scripts/update_golden.sh
                 $<TARGET_FILE:coast_compare>)
