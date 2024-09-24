# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

project(Kokkos-mpi)
# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
# You need this for using Kokkos_ROOT variable
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} includes/)

add_library(lulesh_libs
    lulesh-comm.cc
    lulesh.h
    lulesh-init.cc
    lulesh_tuple.h
    lulesh-util.cc
    lulesh-viz.cc
    includes/cycle.h
    includes/Timer.hxx
    )
set_target_properties(lulesh_libs PROPERTIES CXX_STANDARD 17)
add_executable(lulesh lulesh.cc)
set_target_properties(lulesh PROPERTIES CUDA_ARCHITECTURES "70" CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w -DUSE_MPI=1")
# This is the only thing required to set up compiler/linker flags
target_link_libraries(lulesh_libs Kokkos::kokkos)
target_link_libraries(lulesh Kokkos::kokkos ${MPI_LIBRARIES} lulesh_libs)
