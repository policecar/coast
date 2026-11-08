add_executable(test_campaign test_campaign.cpp)
target_link_libraries(test_campaign PRIVATE doctest::doctest coast_lib)
add_test(NAME unit.campaign.meta_loader COMMAND test_campaign
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
