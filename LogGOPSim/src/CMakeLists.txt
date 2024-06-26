cmake_minimum_required(VERSION 3.12)
project(LogGOPSim LANGUAGES C CXX Fortran)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

# If several versions of a package provide CMake packages, try to use the latest one.
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

include(${CMAKE_MODULE_PATH}/gengetopt.cmake)
include(${CMAKE_MODULE_PATH}/re2c.cmake)
find_gengetopt()
find_re2c()

# build loggopsim
find_package(Graphviz REQUIRED)
include_directories(${GRAPHVIZ_INCLUDE_DIRS})
add_gengetopt_files(LogGOPSim/loggopsim_cmdline)
add_executable(LogGOPSim LogGOPSim/loggopsim_cmdline.c LogGOPSim/LogGOPSim.cpp)
target_link_libraries(LogGOPSim ${GRAPHVIZ_CGRAPH_LIBRARY})

# build txt2bin
add_gengetopt_files(LogGOPSim/txt2bin_cmdline)
add_re2c_files(LogGOPSim/txt2bin)
add_executable(txt2bin LogGOPSim/txt2bin.cpp LogGOPSim/txt2bin_cmdline.c)

# build Schedgen1 (Schedgen2 is pure python)
add_gengetopt_files(Schedgen/schedgen_cmdline)
add_executable(schedgen Schedgen/buffer_element.cpp Schedgen/schedgen.cpp Schedgen/process_trace.cpp Schedgen/schedgen_cmdline.c)

# build liballprof1
find_package(MPI)
include(FortranCInterface)
FortranCInterface_VERIFY()
FortranCInterface_HEADER(fc_mangle.h)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_custom_command(OUTPUT lap1_mpi_c_wrapper.c COMMAND ${PROJECT_SOURCE_DIR}/liballprof/gencode.py c   > lap1_mpi_c_wrapper.c  DEPENDS ${PROJECT_SOURCE_DIR}/liballprof/mpi_header.h)
add_custom_command(OUTPUT lap1_mpi_f_wrapper.c COMMAND ${PROJECT_SOURCE_DIR}/liballprof/gencode.py f77 > lap1_mpi_f_wrapper.c  DEPENDS ${PROJECT_SOURCE_DIR}/liballprof/mpi_header.h)
#include_directories(liballprof)
#add_library(mpipclog1 lap1_mpi_c_wrapper.c liballprof/sync.c)
#add_library(mpipflog1 lap1_mpi_f_wrapper.c liballprof/sync.c)

# build liballprof2
find_package(Clang)
find_package(Unwind)
add_custom_target(mpi_header_avail         DEPENDS ${PROJECT_SOURCE_DIR}/liballprof2/mpi.h)
add_custom_target(mpi_semantics_avail      DEPENDS mpi_sem.yml)
add_custom_target(mpi_wrapper_generated    DEPENDS mpi_c_wrapper.c mpi_f_wrapper.c )
add_custom_command(OUTPUT mpi_c_wrapper.c  mpi_f_wrapper.c COMMAND ${PROJECT_SOURCE_DIR}/liballprof2/gencode.py -s mpi_sem.yml                                                           DEPENDS mpi_semantics_avail)
add_custom_command(OUTPUT mpi_sem.yml                      COMMAND ${PROJECT_SOURCE_DIR}/liballprof2/gensem.py -l ${CLANG_INSTALL_PREFIX}/lib -m ${PROJECT_SOURCE_DIR}/liballprof2/mpi.h DEPENDS mpi_header_avail)
add_library(mpipclog SHARED mpi_c_wrapper.c)
add_library(mpipflog SHARED mpi_f_wrapper.c)
add_dependencies(mpipclog mpi_wrapper_generated)
add_dependencies(mpipflog mpi_wrapper_generated)
target_link_libraries(mpipclog ${MPI_C_LIBRARIES})
target_link_libraries(mpipflog ${MPI_Fortran_LIBRARIES})
if (Unwind_FOUND)
  target_link_libraries(mpipclog unwind::unwind)
  target_link_libraries(mpipflog unwind::unwind)
endif()

include(CTest)
# test the c++ toolchain for some pattern
add_test(NAME schedgen_binomialtreebcast COMMAND schedgen -o schedule.goal --commsize 8 --ptrn binomialtreebcast)
add_test(NAME txt2bin_binomialtreebcast COMMAND txt2bin -i schedule.goal -o schedule.bin DEPENDS schedgen_binomialtreebcast)
add_test(NAME loggopsim_binomialtreebcast COMMAND LogGOPSim -f schedule.bin DEPENDS txt2bin_binomialtreebcast)

# test the python toolchain for some pattern
add_test(NAME schedgen2_bcast COMMAND ${PYTHON_EXECUTABLE} schedgen.py bcast --output schedule.goal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Schedgen2)
add_test(NAME txt2bin_schedgen2_bcast COMMAND txt2bin -i ${PROJECT_SOURCE_DIR}/Schedgen2/schedule.goal -o schedule.bin)
add_test(NAME loggopsim_schedgen2_bcast COMMAND LogGOPSim -f schedule.bin)

# test liballprof

# test liballprof2
#c wrapper test
add_executable(test_lap2_c ${PROJECT_SOURCE_DIR}/../tests/mpi_helloworld.c)
target_link_libraries(test_lap2_c mpipclog)
add_test(NAME trace_lap2_c COMMAND ${MPIEXEC_EXECUTABLE} --host localhost:4 ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_lap2_c>)
add_test(NAME lap2_c_trace_exists  COMMAND ${CMAKE_COMMAND} -E cat lap2-trace-rank-1-of-4.txt)

# fortran wrapper test
add_executable(test_lap2_f ${PROJECT_SOURCE_DIR}/../tests/mpi_helloworld.f90)
target_link_libraries(test_lap2_f mpipflog)
add_test(NAME trace_lap2_f COMMAND ${MPIEXEC_EXECUTABLE} --host localhost:4 ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_lap2_f>)
add_test(NAME lap2_f_trace_exists  COMMAND ${CMAKE_COMMAND} -E cat lap2-trace-rank-1-of-4.txt )
