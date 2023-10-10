cmake_minimum_required(VERSION 3.15)

project(CoScheduling)

# Look for an installed Kokkos
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(networkInhib networkInhibitor/networkInhib.cpp)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(networkInhib ${MPI_LIBRARIES})
