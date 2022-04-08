cmake_minimum_required(VERSION 3.18.2)
project(ddptensor VERSION 1.0)

set(LLVM_PATH ${PROJECT_SOURCE_DIR}/mlir-llvm)
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
  set(CONDA_PREFIX UNSET)
endif()

if(DEFINED ENV{MKLROOT})
  set(MKLROOT ENV{MKLROOT})
else()
  set(MKLROOT ${CONDA_PREFIX})
endif()
if(DEFINED ENV{TBBROOT})
  set(TBBROOT ENV{TBBROOT})
else()
  set(TBBROOT ${CONDA_PREFIX})
endif()

if(MKLROOT STREQUAL UNSET OR TBBROOT STREQUAL UNSET)
  message(FATAL_ERROR "MKLROOT and TBBROOT not set (nor CONDA_PREFIX)")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common installation directories
#include(GNUInstallDirs)

# ===============
# Deps
# ===============
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/third_party/mlir-llvm)

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG)
find_package(MPI REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

#find_package(OpenMP)

set(MKL_LIBRARIES -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread -lrt -ldl -lm)
#set(CMAKE_INSTALL_RPATH $ENV{MKLROOT}/lib)
# Use -fPIC even if statically compiled
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(P2C_HPP ${PROJECT_SOURCE_DIR}/src/include/ddptensor/p2c_ids.hpp)
# Generate enums
add_custom_command(
    COMMAND python ${PROJECT_SOURCE_DIR}/scripts/code_gen.py ${PROJECT_SOURCE_DIR}/ddptensor/array_api.py ${P2C_HPP}
    DEPENDS ${PROJECT_SOURCE_DIR}/scripts/code_gen.py ${PROJECT_SOURCE_DIR}/ddptensor/array_api.py
    OUTPUT ${P2C_HPP}
    COMMENT "Generating ${P2C_HPP}."
    )

# ============
# Target
# ============
FILE(GLOB MyCppSources ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/include/ddptensor/*.hpp)
set(MyCppSources ${MyCppSources} ${PROJECT_SOURCE_DIR}/src/jit/mlir.cpp ${P2C_HPP})

pybind11_add_module(_ddptensor MODULE ${MyCppSources})

target_compile_definitions(_ddptensor PRIVATE XTENSOR_USE_XSIMD=1 XTENSOR_USE_TBB=1 USE_MKL=1 DDPT_2TYPES=1)
target_include_directories(_ddptensor PRIVATE
  ${PROJECT_SOURCE_DIR}/src/include
  ${PROJECT_SOURCE_DIR}/third_party/xtl/include
  ${PROJECT_SOURCE_DIR}/third_party/xsimd/include
  ${PROJECT_SOURCE_DIR}/third_party/xtensor-blas/include
  ${PROJECT_SOURCE_DIR}/third_party/xtensor/include
  ${PROJECT_SOURCE_DIR}/third_party/bitsery/include
  ${MPI_INCLUDE_PATH}
  $ENV{MKLROOT}/include
  $ENV{TBBROOT}/include
  ${pybind11_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

#target_compile_options(_ddptensor PRIVATE -fopenmp)
target_link_libraries(_ddptensor PRIVATE
    ${MPI_C_LIBRARIES}
    ${MKL_LIBRARIES}
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMTarget
    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRTransforms
    MLIRFuncTransforms
    MLIRLinalgTransforms
    MLIRLinalgToLLVM
    MLIRMathToLLVM
    MLIRMathToLibm
    MLIRTensorTransforms
    MLIRReconcileUnrealizedCasts)
