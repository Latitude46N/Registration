cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(test)
#find packages
find_package(PCL 1.8 REQUIRED)
#include directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(test main.cpp)
#
target_link_libraries(test ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_SEARCH_LIBRARIES}  ${PCL_FEATURES_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES})

