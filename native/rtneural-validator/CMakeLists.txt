cmake_minimum_required(VERSION 3.20)

project(rtneural-validator VERSION 0.1.0 LANGUAGES CXX)

add_executable(rtneural-validator src/main.cpp)
target_compile_features(rtneural-validator PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(rtneural-validator PRIVATE /W4)
else()
  target_compile_options(rtneural-validator PRIVATE -Wall -Wextra -Wpedantic)
endif()
