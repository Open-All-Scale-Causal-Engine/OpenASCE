cmake_minimum_required(VERSION 3.9.4)

project(gbct_utils LANGUAGES CXX)

# SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"' -DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF=1 -DPYBIND11_NO_GIL_CHECKS=1")
execute_process(COMMAND python -m pybind11 --cmakedir RESULT_VARIABLE tmp OUTPUT_VARIABLE PYBIND11_CMAKE OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PYBIND11_CMAKE})

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(root_path ${CMAKE_SOURCE_DIR})
get_filename_component(CPP_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
set(external_libs_path "${PROJECT_DIR}/../external_libs")
include_directories(${root_path})
include_directories(${external_libs_path})

set(pybind11_DIR ${PYBIND11_CMAKE})
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED)
if (Python3_FOUND)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ${PYBIND11_VAR}")

FILE(GLOB src_cpps "${root_path}/src/*.cpp" "${root_path}/src/*.hpp" "include/*.hpp" "include/utils/*.hpp")
FILE(GLOB src_hs "include/*.h" "include/utils/*.h")

set(gbct_files ${src_cpps} ${src_hs})

message("gbct files ${gbct_files}")
pybind11_add_module(gbct_utils MODULE ${gbct_files})

if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET gbct_utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

target_compile_definitions(gbct_utils
                           PRIVATE VERSION=${EXAMPLE_VERSION_INFO})


