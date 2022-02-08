import os
from os.path import join as jp
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

mpiroot = os.environ.get('MPIROOT')
mklroot = os.environ.get('MKLROOT')
xtroot = os.getenv('XTROOT', 'third_party')

xt_includes = [jp(xtroot, x, "include") for x in ("xtl", "xsimd", "xtensor-blas", "xtensor")]

ext_modules = [
    Pybind11Extension(
        "ddptensor._ddptensor",
        glob("src/*.cpp"),
        include_dirs = xt_includes + [jp(mpiroot, "include"), jp("third_party", "bitsery", "include"), jp("src", "include"), ],
        extra_compile_args = ["-DUSE_MKL", "-DXTENSOR_USE_XSIMD=1", "-DXTENSOR_USE_OPENMP=1",
                              "-std=c++17", "-fopenmp",
                              "-Wno-unused-but-set-variable", "-Wno-sign-compare", "-Wno-unused-local-typedefs", "-Wno-reorder",
                              "-march=native", "-O0", "-g"],
        libraries = ["mpi", "mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5", "pthread", "rt", "dl", "m"],
        library_dirs = [jp(mpiroot, "lib")],
        language = 'c++'
    ),
]

setup(name="ddptensor",
      version="0.1",
      description="Distributed Tensor and more",
      packages=["ddptensor", "ddptensor.numpy", "ddptensor.torch"],
      ext_modules=ext_modules
)
