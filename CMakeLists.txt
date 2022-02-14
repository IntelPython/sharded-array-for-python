cmake_minimum_required(VERSION 3.18.2)
project(ddptensor VERSION 1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common installation directories
#include(GNUInstallDirs)

# Use -fPIC even if statically compiled
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============
# Target
# ============
FILE(GLOB MyCppSources ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/include/ddptensor/*.hpp)

# Create the mymath library
add_library(_ddptensor SHARED ${MyCppSources})

target_include_directories(_ddptensor PRIVATE ${PROJECT_SOURCE_DIR}/src/include ${PROJECT_SOURCE_DIR}/third_party/xtl/include ${PROJECT_SOURCE_DIR}/third_party/xsimd/include ${PROJECT_SOURCE_DIR}/third_party/xtensor-blas/include ${PROJECT_SOURCE_DIR}/third_party/xtensor/include ${PROJECT_SOURCE_DIR}/third_party/bitsery/include)

# ===============
# Deps
# ===============

# Find Python3 and NumPy
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

find_package(MPI REQUIRED)
find_package(pybind11 CONFIG)
include_directories(SYSTEM ${MPI_INCLUDE_PATH} ${pybind11_INCLUDE_DIRS})
target_link_libraries(_ddptensor ${MPI_C_LIBRARIES})
