cmake_minimum_required(VERSION 3.16)
project(lab_2_mpi)

set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/include/x86_64-linux-gnu/mpich/)
add_executable(lab_2_mpi main.cpp)
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
