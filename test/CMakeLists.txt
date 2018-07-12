cmake_minimum_required(VERSION 2.8)

# Project
project(nimu-ann-test)

#--------------------------------------------------------------------------------------------------
# Test files
set(nimu_ann_test_SOURCES
  ann_static.cpp
)

#--------------------------------------------------------------------------------------------------
# Setup testing
enable_testing()
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

# Test executable
add_executable(nimu-ann-test ${nimu_ann_test_SOURCES})
target_link_libraries(nimu-ann-test Boost::unit_test_framework)
target_include_directories(nimu-ann-test PUBLIC ../include)
target_compile_definitions(nimu-ann-test PRIVATE BOOST_TEST_DYN_LINK)

# Test directive 
add_test(NAME nimu-ann-test-ann-static COMMAND nimu-ann-test-ann-static)

#--------------------------------------------------------------------------------------------------
# Install
install(TARGETS nimu-ann-test DESTINATION bin)


