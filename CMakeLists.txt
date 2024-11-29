cmake_minimum_required(VERSION 3.18.2)
project(sharpy VERSION 1.0)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
  set(CONDA_PREFIX UNSET)
endif()
if(DEFINED ENV{MLIRROOT})
  set(MLIR_ROOT $ENV{MLIRROOT})
else()
  set(MLIR_ROOT ${CONDA_PREFIX})
endif()
set(LLVM_ROOT ${MLIR_ROOT})
if(DEFINED ENV{IMEXROOT})
  set(IMEX_ROOT $ENV{IMEXROOT})
else()
  set(IMEX_ROOT ${CONDA_PREFIX})
endif()
if(DEFINED ENV{ZLIBROOT})
  set(ZLIB_ROOT $ENV{ZLIBROOT})
else()
  set(ZLIB_ROOT ${CONDA_PREFIX})
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expected LLVM SHA
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/imex_version.txt EXPECTED_IMEX_SHA)
message(STATUS "Expected IMEX sha: \"${EXPECTED_IMEX_SHA}\"")

# Common installation directories
#include(GNUInstallDirs)

# ===============
# Deps
# ===============
# Find dependent packages like IMEX, Python3 and NumPy
# find_package(ZLIB REQUIRED)
find_library(LIBZ z NAMES libz.so libz.so.1 REQUIRED PATHS ${ZLIB_ROOT}/lib)
message("FOUND zlib ${LIBZ}")
set(ZLIB_LIBRARY ${LIBZ})
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
# FindMPI does not work with iMPI conda packages
set(MPI_INCLUDE_PATH $ENV{I_MPI_ROOT}/include)
set(MPI_CXX_LIBRARIES $ENV{I_MPI_ROOT}/lib/libmpi.so $ENV{I_MPI_ROOT}/lib/libmpicxx.so)
# find_package(MPI REQUIRED CXX)
find_package(LLVM REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(IMEX REQUIRED CONFIG REQUIRED)
message(STATUS "Using IMEXConfig.cmake in: ${IMEX_DIR}")

list(APPEND CMAKE_MODULE_PATH "${IMEX_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
#include(AddLLVM)
#include(AddMLIR)
#include(AddIMEX)

# LLVM defs, most impoprtatntly __STDC_FORMAT_MACROS for PRIxPTR et al
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS})
# macro for mlir/imex root directories
add_compile_definitions(CMAKE_MLIR_ROOT="${MLIR_ROOT}" CMAKE_IMEX_ROOT="${IMEX_ROOT}" FORTIFY_SOURCE=2)
add_compile_options(
    "-ftemplate-backtrace-limit=0"
    "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},GNU>:-flarge-source-files>"
    "-Wall" "-Wformat" "-fPIC" "-Wno-comment"
    "$<$<CONFIG:DEBUG>:-fstack-protector>"
    "$<$<CONFIG:RELEASE>:-flto>"
    "$<$<CONFIG:RELEASE>:-z>"
    "$<$<CONFIG:RELEASE>:noexecstack>"
    "$<$<CONFIG:RELEASE>:-fstack-protector-strong>"
)
add_link_options("-pie" "$<$<CONFIG:RELEASE>:-Wl,-z,relro,-z,now>")

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
# we assume the typical python site-packages layout
SET(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../../..")

#find_package(OpenMP)

# set(MKL_LIBRARIES -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread -lrt -ldl -lm)
# Use -fPIC even if statically compiled
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(P2C_HPP ${PROJECT_SOURCE_DIR}/src/include/sharpy/p2c_ids.hpp)
# Generate enums
add_custom_command(
    COMMAND python ${PROJECT_SOURCE_DIR}/scripts/code_gen.py ${PROJECT_SOURCE_DIR}/sharpy/array_api.py ${P2C_HPP}
    DEPENDS ${PROJECT_SOURCE_DIR}/scripts/code_gen.py ${PROJECT_SOURCE_DIR}/sharpy/array_api.py
    OUTPUT ${P2C_HPP}
    COMMENT "Generating ${P2C_HPP}."
    )

# ============
# Target
# ============

FILE(GLOB Hpps ${PROJECT_SOURCE_DIR}/src/include/sharpy/*.hpp)
FILE(GLOB JitHpps ${PROJECT_SOURCE_DIR}/src/include/sharpy/jit/*.hpp)
set(Hpps ${Hpps} ${JitHpps} ${P2C_HPP})

set(SHARPYSrcs
    ${PROJECT_SOURCE_DIR}/src/_sharpy.cpp
    ${PROJECT_SOURCE_DIR}/src/BodyBuilder.cpp
    ${PROJECT_SOURCE_DIR}/src/Creator.cpp
    ${PROJECT_SOURCE_DIR}/src/EWBinOp.cpp
    ${PROJECT_SOURCE_DIR}/src/EWUnyOp.cpp
    ${PROJECT_SOURCE_DIR}/src/IO.cpp
    ${PROJECT_SOURCE_DIR}/src/LinAlgOp.cpp
    ${PROJECT_SOURCE_DIR}/src/ManipOp.cpp
    ${PROJECT_SOURCE_DIR}/src/Random.cpp
    ${PROJECT_SOURCE_DIR}/src/ReduceOp.cpp
    ${PROJECT_SOURCE_DIR}/src/SetGetItem.cpp
    ${PROJECT_SOURCE_DIR}/src/jit/mlir.cpp
    ${PROJECT_SOURCE_DIR}/src/Deferred.cpp
    ${PROJECT_SOURCE_DIR}/src/Service.cpp
    ${PROJECT_SOURCE_DIR}/src/Mediator.cpp
    ${PROJECT_SOURCE_DIR}/src/MPIMediator.cpp
    ${PROJECT_SOURCE_DIR}/src/CollComm.cpp
    ${PROJECT_SOURCE_DIR}/src/NDArray.cpp
    ${PROJECT_SOURCE_DIR}/src/Factory.cpp
    ${PROJECT_SOURCE_DIR}/src/Registry.cpp
    ${PROJECT_SOURCE_DIR}/src/_deferred.cpp
)
set(IDTRSrcs
    ${PROJECT_SOURCE_DIR}/src/idtr.cpp
    ${PROJECT_SOURCE_DIR}/src/MPITransceiver.cpp
    ${PROJECT_SOURCE_DIR}/src/Transceiver.cpp
)

pybind11_add_module(_sharpy ${SHARPYSrcs} ${Hpps})
add_library(idtr SHARED ${IDTRSrcs} ${Hpps})
set(AllTargets _sharpy idtr)

include_directories(
  ${PROJECT_SOURCE_DIR}/src/include
  ${PROJECT_SOURCE_DIR}/third_party/bitsery/include
  ${MPI_INCLUDE_PATH}
  ${pybind11_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${IMEX_INCLUDE_DIRS})

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_options(_sharpy PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/export.txt")
  # target_link_options(idtr PRIVATE "LINKER:-fvisibility=hidden" "LINKER:--exclude-libs,All")
endif()

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(mlir_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(imex_dialect_libs GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(imex_conversion_libs GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
get_property(imex_extension_libs GLOBAL PROPERTY IMEX_EXTENSION_LIBS)
get_property(imex_translation_libs GLOBAL PROPERTY IMEX_TRANSLATION_LIBS)

target_link_directories(_sharpy PRIVATE ${CONDA_PREFIX}/lib ${IMEX_ROOT}/lib )
target_link_directories(idtr PRIVATE ${CONDA_PREFIX}/lib)

target_link_libraries(_sharpy PRIVATE
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
    ${mlir_translation_libs}
    MLIROptLib
    MLIRExecutionEngine
    MLIRCopyOpInterface
    ${imex_dialect_libs}
    ${imex_conversion_libs}
    ${imex_extension_libs}
    ${imex_translation_libs}
    IMEXTransforms
    IMEXUtil
    LLVMX86CodeGen
    LLVMX86AsmParser
    idtr
    TBB::tbb
    ${LIBZ}
)
target_link_libraries(idtr PRIVATE
    ${MPI_CXX_LIBRARIES}
    TBB::tbb
)
