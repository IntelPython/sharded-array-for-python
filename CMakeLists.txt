cmake_minimum_required(VERSION 3.18.2)
project(sharpy VERSION 1.0)

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
  set(CONDA_PREFIX UNSET)
endif()

if(DEFINED ENV{MKLROOT})
  set(MKL_ROOT $ENV{MKLROOT})
else()
  set(MKL_ROOT ${CONDA_PREFIX})
endif()
if(DEFINED ENV{TBBROOT})
  set(TBB_ROOT $ENV{TBBROOT})
else()
  set(TBB_ROOT ${CONDA_PREFIX})
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
message("ll ${IMEX_ROOT}")
if(MKLROOT STREQUAL UNSET OR TBBROOT STREQUAL UNSET)
  message(FATAL_ERROR "MKLROOT and TBBROOT not set (nor CONDA_PREFIX)")
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
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG)
find_package(MPI REQUIRED)
find_package(LLVM REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(IMEX REQUIRED CONFIG)
message(STATUS "Using IMEXConfig.cmake in: ${IMEX_DIR}")

list(APPEND CMAKE_MODULE_PATH "${IMEX_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
#include(AddLLVM)
#include(AddMLIR)
#include(AddIMEX)

# macro for mlir root directory
add_compile_definitions(CMAKE_MLIR_ROOT="${MLIR_ROOT}" CMAKE_IMEX_ROOT="${IMEX_ROOT}")

#find_package(OpenMP)

set(MKL_LIBRARIES -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread -lrt -ldl -lm)
#set(CMAKE_INSTALL_RPATH $ENV{MKLROOT}/lib)
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
    ${PROJECT_SOURCE_DIR}/src/Creator.cpp
    ${PROJECT_SOURCE_DIR}/src/EWBinOp.cpp
    ${PROJECT_SOURCE_DIR}/src/EWUnyOp.cpp
    ${PROJECT_SOURCE_DIR}/src/IEWBinOp.cpp
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

pybind11_add_module(_sharpy MODULE ${SHARPYSrcs} ${Hpps})
add_library(idtr SHARED ${IDTRSrcs} ${Hpps})
set(AllTargets _sharpy idtr)

add_compile_definitions(USE_MKL=1)
add_compile_options("-ftemplate-backtrace-limit=0")
include_directories(
  ${PROJECT_SOURCE_DIR}/src/include
  ${PROJECT_SOURCE_DIR}/third_party/bitsery/include
  ${MPI_INCLUDE_PATH}
  $ENV{MKLROOT}/include
  $ENV{TBBROOT}/include
  ${pybind11_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${IMEX_INCLUDE_DIRS})

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_options(_sharpy PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/export.txt")
  # target_link_options(idtr PRIVATE "LINKER:-fvisibility=hidden" "LINKER:--exclude-libs,All")
endif()

#compile_options(_sharpy PRIVATE -fopenmp)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(mlir_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(imex_dialect_libs GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(imex_conversion_libs GLOBAL PROPERTY IMEX_CONVERSION_LIBS)

#llvm_update_compile_flags(_sharpy)
target_link_directories(_sharpy PRIVATE ${CONDA_PREFIX}/lib ${IMEX_ROOT}/lib)
target_link_directories(idtr PRIVATE ${CONDA_PREFIX}/lib)

target_link_libraries(_sharpy PRIVATE
    # ${MKL_LIBRARIES}
    # tbb
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
    ${mlir_translation_libs}
    MLIROptLib
    MLIRExecutionEngine
    ${imex_dialect_libs}
    ${imex_conversion_libs}
    IMEXTransforms
    IMEXUtil
    LLVMX86CodeGen
    LLVMX86AsmParser
    idtr
    tbb
)
target_link_libraries(idtr PRIVATE
    ${MPI_CXX_LIBRARIES}
    # ${MKL_LIBRARIES}
    tbb
)
