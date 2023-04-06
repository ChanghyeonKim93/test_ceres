cmake_minimum_required(VERSION 3.9)
project(test_ceres)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -o2")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(cmake_modules)
find_package(Ceres REQUIRED)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

include_directories( ${INCLUDE_DIRS} )

add_executable (test_include test_include.cpp ${SRC})
target_link_libraries(test_include PUBLIC ${CERES_LIBRARIES})
target_include_directories(test_include PUBLIC ${CERES_INCLUDE_DIRS})


add_executable (test_optimize_simple test_optimize_simple.cpp ${SRC})
target_link_libraries(test_optimize_simple PUBLIC ${CERES_LIBRARIES})
target_include_directories(test_optimize_simple PUBLIC ${CERES_INCLUDE_DIRS})