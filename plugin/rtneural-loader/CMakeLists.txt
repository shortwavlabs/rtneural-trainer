cmake_minimum_required(VERSION 3.22)

project(RTNeuralLoaderPlugin VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS deployment target")

option(RTNEURAL_LOADER_COPY_PLUGIN_AFTER_BUILD "Copy plugin to the local plugin folder after build" ON)

set(RTNEURAL_EIGEN ON CACHE BOOL "Use RTNeural Eigen backend" FORCE)
set(RTNEURAL_STL OFF CACHE BOOL "Use RTNeural STL backend" FORCE)
set(RTNEURAL_XSIMD OFF CACHE BOOL "Use RTNeural xsimd backend" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Do not build RTNeural tests" FORCE)
set(BUILD_BENCH OFF CACHE BOOL "Do not build RTNeural benchmarks" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build RTNeural examples" FORCE)

set(RTNEURAL_DEFAULT_LOCAL_PATH "/Users/shortwavlabs/Workspace/rt-neural/RTNeural")
if(DEFINED ENV{RTNEURAL_LOCAL_PATH})
    set(RTNEURAL_DEFAULT_LOCAL_PATH "$ENV{RTNEURAL_LOCAL_PATH}")
endif()

set(RTNEURAL_LOCAL_PATH "${RTNEURAL_DEFAULT_LOCAL_PATH}" CACHE PATH "Local RTNeural checkout")
if(EXISTS "${RTNEURAL_LOCAL_PATH}/CMakeLists.txt")
    message(STATUS "rtneural-loader -- Using local RTNeural checkout: ${RTNEURAL_LOCAL_PATH}")
    set(RTNEURAL_SOURCE_PATH "${RTNEURAL_LOCAL_PATH}")
    add_subdirectory("${RTNEURAL_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/rtneural-local-build" EXCLUDE_FROM_ALL)
else()
    message(STATUS "rtneural-loader -- Local RTNeural checkout not found, fetching shortwavlabs/rtneural-extended")
    FetchContent_Declare(
        RTNeural
        GIT_REPOSITORY https://github.com/shortwavlabs/rtneural-extended.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(RTNeural)
    set(RTNEURAL_SOURCE_PATH "${rtneural_SOURCE_DIR}")
endif()

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.12
)
FetchContent_MakeAvailable(JUCE)

if(TARGET RTNeural)
    if(MSVC)
        target_compile_options(RTNeural PRIVATE /w)
    else()
        target_compile_options(RTNeural PRIVATE -w)
    endif()
endif()

juce_add_plugin(RTNeuralLoader
    VERSION 0.1.0
    COMPANY_NAME "Shortwav Labs"
    PRODUCT_NAME "RTNeural Loader"
    BUNDLE_ID "labs.shortwav.rtneural-loader"
    PLUGIN_MANUFACTURER_CODE SwLv
    PLUGIN_CODE RtL1
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD ${RTNEURAL_LOADER_COPY_PLUGIN_AFTER_BUILD}
    FORMATS AU VST3 Standalone
)

juce_generate_juce_header(RTNeuralLoader)

target_sources(RTNeuralLoader
    PRIVATE
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
)

target_include_directories(RTNeuralLoader
    SYSTEM PRIVATE
        "${RTNEURAL_SOURCE_PATH}"
)

target_compile_definitions(RTNeuralLoader
    PUBLIC
        JUCE_REPORT_APP_USAGE=0
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        RTNEURAL_NO_DEBUG=1
)

target_link_libraries(RTNeuralLoader
    PRIVATE
        RTNeural
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
