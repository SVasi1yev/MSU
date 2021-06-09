cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

set(MPI_CXX_COMPILER /opt/ibm/spectrum_mpi/bin/mpic++)

find_package(Torch REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(MPI)

include_directories(${MPI_INCLUDE_PATH})

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main "${MPI_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 11)