cmake_minimum_required(VERSION 3.5)
project(spectral_density)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Build the Python module from both py_module.cpp and main.cpp.
add_library(spectral_density MODULE py_module.cpp main.cpp)
target_compile_definitions(spectral_density PRIVATE BUILD_PYTHON)
target_link_libraries(spectral_density PRIVATE pybind11::module Eigen3::Eigen)
set_target_properties(spectral_density PROPERTIES PREFIX "" SUFFIX ".so")

# Standalone executable target (builds main.cpp without BUILD_PYTHON defined).
add_executable(spectral_density_exec main.cpp)
target_link_libraries(spectral_density_exec PRIVATE Eigen3::Eigen)
