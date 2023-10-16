cmake_minimum_required(VERSION 3.15)

project(CoScheduling)

# Look for an installed Kokkos
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

add_executable(networkInhib networkInhibitor/networkInhib.cpp)
add_executable(pingPong pingPong/pingPong.cpp)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(networkInhib ${MPI_LIBRARIES})
target_link_libraries(pingPong ${MPI_LIBRARIES})