cmake_minimum_required(VERSION 3.8)
project(Segmentator VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

file(GLOB_RECURSE CPP_FILES CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.hpp)
list(FILTER CPP_FILES EXCLUDE REGEX ".*app\\.cpp$")

# lib
add_library(${PROJECT_NAME} ${CPP_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${OpenCV_LIB_PATH})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# app
add_executable(${PROJECT_NAME}App src/app.cpp)
target_include_directories(${PROJECT_NAME}App PRIVATE $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${OpenCV_INCLUDE_DIRS} ${EZPARSER_INLCUDE_DIR})
target_link_directories(${PROJECT_NAME}App PUBLIC ${OpenCV_LIB_PATH})
target_link_libraries(${PROJECT_NAME}App ${PROJECT_NAME} ${OpenCV_LIBS})

add_custom_command(
    TARGET ${PROJECT_NAME}App POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Res/haarcascade_eye_tree_eyeglasses.xml $<TARGET_FILE_DIR:Segmentator>
)
# configure_file( ${CMAKE_CURRENT_BINARY_DIR}/haarcascade_eye_tree_eyeglasses.xml COPYONLY)