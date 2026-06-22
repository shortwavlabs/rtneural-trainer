cmake_minimum_required(VERSION 3.20)

project(rtneural-validator VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural STL backend" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Do not build RTNeural tests" FORCE)
set(BUILD_BENCH OFF CACHE BOOL "Do not build RTNeural benchmarks" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build RTNeural examples" FORCE)

set(RTNEURAL_DEFAULT_LOCAL_PATH "/Users/shortwavlabs/Workspace/rt-neural/RTNeural")
if(DEFINED ENV{RTNEURAL_LOCAL_PATH})
    set(RTNEURAL_DEFAULT_LOCAL_PATH "$ENV{RTNEURAL_LOCAL_PATH}")
endif()

set(RTNEURAL_LOCAL_PATH "${RTNEURAL_DEFAULT_LOCAL_PATH}" CACHE PATH "Local RTNeural checkout")
if(EXISTS "${RTNEURAL_LOCAL_PATH}/CMakeLists.txt")
    message(STATUS "rtneural-validator -- Using local RTNeural checkout: ${RTNEURAL_LOCAL_PATH}")
    set(RTNEURAL_SOURCE_PATH "${RTNEURAL_LOCAL_PATH}")
    add_subdirectory("${RTNEURAL_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/rtneural-local-build" EXCLUDE_FROM_ALL)
else()
    message(STATUS "rtneural-validator -- Local RTNeural checkout not found, fetching pinned RTNeural")
    FetchContent_Declare(
        RTNeural
        GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
        GIT_TAG 1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d
    )
    FetchContent_MakeAvailable(RTNeural)
    set(RTNEURAL_SOURCE_PATH "${rtneural_SOURCE_DIR}")
endif()

if(TARGET RTNeural)
    # Keep warnings from the pinned third-party dependency out of CI logs.
    if(MSVC)
        target_compile_options(RTNeural PRIVATE /w)
    else()
        target_compile_options(RTNeural PRIVATE -w)
    endif()
endif()

add_executable(rtneural-validator src/main.cpp)
target_compile_features(rtneural-validator PRIVATE cxx_std_17)
if(CMAKE_BUILD_TYPE)
  set(RTNEURAL_VALIDATOR_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
else()
  set(RTNEURAL_VALIDATOR_BUILD_TYPE "unknown")
endif()
target_compile_definitions(
  rtneural-validator
  PRIVATE
    RTNEURAL_NO_DEBUG=1
    RTNEURAL_VALIDATOR_BACKEND="rtneural-stl"
    RTNEURAL_VALIDATOR_BUILD_TYPE="${RTNEURAL_VALIDATOR_BUILD_TYPE}"
)
target_include_directories(rtneural-validator SYSTEM PRIVATE "${RTNEURAL_SOURCE_PATH}")
target_link_libraries(rtneural-validator PRIVATE RTNeural)

if(MSVC)
  target_compile_options(rtneural-validator PRIVATE /W4)
else()
  target_compile_options(rtneural-validator PRIVATE -Wall -Wextra -Wpedantic)
endif()
