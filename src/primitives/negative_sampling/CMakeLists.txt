cmake_minimum_required(VERSION 3.4...3.18)
project(negative_sampling)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Assuming the path is /opt/miniconda3/envs/rel-mm/lib/python3.10/site-packages/pybind11/include
set(CMAKE_PREFIX_PATH "/opt/miniconda3/envs/rel-mm/lib/python3.10/site-packages/pybind11/include" ${CMAKE_PREFIX_PATH})
set(pybind11_DIR "/opt/miniconda3/envs/rel-mm/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(negative_sampling negative_sampling.cpp)