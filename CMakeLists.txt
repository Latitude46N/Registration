cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(RGSTRN)
#find packages
find_package(PCL 1.8 REQUIRED)
#include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(Registration main.cpp registration.cpp)
#
target_link_libraries(Registration ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES})

