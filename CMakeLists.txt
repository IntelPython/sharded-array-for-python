cmake_minimum_required(VERSION 3.18.2)
project(ddptensor VERSION 1.0)

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

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG)
find_package(MPI REQUIRED)

set(MKL_LIBRARIES -L$ENV{MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lrt -ldl -lm)
# Use -fPIC even if statically compiled
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============
# Target
# ============
FILE(GLOB MyCppSources ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/include/ddptensor/*.hpp)

# Create the mymath library
#add_library(_ddptensor MODULE ${MyCppSources})
pybind11_add_module(_ddptensor THIN_LTO ${MyCppSources})

target_compile_definitions(_ddptensor PRIVATE USE_MKL=1 XTENSOR_USE_XSIMD=1 XTENSOR_USE_OPENMP=1) # DDPT_2TYPES=1)
target_include_directories(_ddptensor PRIVATE ${PROJECT_SOURCE_DIR}/src/include ${PROJECT_SOURCE_DIR}/third_party/xtl/include ${PROJECT_SOURCE_DIR}/third_party/xsimd/include ${PROJECT_SOURCE_DIR}/third_party/xtensor-blas/include ${PROJECT_SOURCE_DIR}/third_party/xtensor/include ${PROJECT_SOURCE_DIR}/third_party/bitsery/include ${MPI_INCLUDE_PATH} $ENV{MKLROOT}/include ${pybind11_INCLUDE_DIRS})
target_link_libraries(_ddptensor PRIVATE ${MPI_C_LIBRARIES} ${MKL_LIBRARIES})
