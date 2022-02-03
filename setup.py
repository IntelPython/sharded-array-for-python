import os
from os.path import join as jp
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

mpiroot = os.environ.get('MPIROOT')

ext_modules = [
    Pybind11Extension(
        "ddptensor._ddptensor",
        glob("src/*.cpp"),
        include_dirs=[jp(mpiroot, "include"), jp("third_party", "bitsery", "include"), jp("src", "include"), ],
        extra_compile_args=["-DUSE_MKL", "-std=c++17", "-Wno-unused-but-set-variable", "-Wno-sign-compare", "-Wno-unused-local-typedefs", "-Wno-reorder", "-O0", "-g"],
        libraries=["mpi", "rt", "pthread", "dl", "mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5", "m"],
        library_dirs=[jp(mpiroot, "lib")],
        language='c++'
    ),
]

setup(name="ddptensor",
      version="0.1",
      description="Distributed Tensor and more",
      packages=["ddptensor", "ddptensor.numpy", "ddptensor.torch"],
      ext_modules=ext_modules
)
