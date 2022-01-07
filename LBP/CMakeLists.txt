cmake_minimum_required(VERSION 3.8)
project(LBP VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.hpp)
add_executable(${PROJECT_NAME} ${CPP_FILES})


target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS} Segmentator)

target_link_directories(${PROJECT_NAME} PUBLIC ${OpenCV_LIB_PATH})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Segmentator)