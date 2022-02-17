cmake_minimum_required(VERSION 3.8)
project(iris_recognition VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

set(EZPARSER_INLCUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Ext/ezOptionParser)

if(NOT APPLE)
    add_compile_definitions(-DUSE_PARALLEL_ALGORITHMS)
endif(APPLE)

add_subdirectory(Segmentator)