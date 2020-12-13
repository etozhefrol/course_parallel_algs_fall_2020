# cmake_minimum_required(VERSION <specify CMake version here>)
project(1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
include_directories(/usr/include/mpi/)
add_executable(1 main.cpp)