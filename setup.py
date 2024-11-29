import multiprocessing
import os
import pathlib
import subprocess
from os.path import join as jp

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    # don't invoke the original build_ext for this special extension
    def __init__(self, name, cmake_lists_dir=".", **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = os.path.abspath(
            os.path.dirname(jp(self.build_lib, self.get_ext_fullname(ext.name)))
        )
        # example of cmake args
        config = "Debug"  # if self.debug else 'RelWithDebInfo' #'Release'

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            # f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_BUILD_TYPE={config}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
            "-G=Ninja",
            '-DCMAKE_CXX_FLAGS="-fuse-ld=lld"',
            "-DLLVM_USE_LINKER=lld",
            "-DLLVM_ENABLE_LLD=ON",
            f"-DCMAKE_PREFIX_PATH={os.getenv('CONDA_PREFIX')}/lib/cmake",
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if not os.path.exists(extdir):
            os.makedirs(extdir)

        # Config
        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
        )

        # Build
        subprocess.check_call(
            [
                "cmake",
                "--build",
                ".",
                "--config",
                config,
                f"-j{multiprocessing.cpu_count()}",
            ],
            cwd=self.build_temp,
        )

        os.chdir(str(cwd))


setup(
    name="sharpy",
    version="0.2",
    description="Distributed array and more",
    packages=["sharpy", "sharpy.numpy", "sharpy.random"],  # "sharpy.torch"],
    ext_modules=[CMakeExtension("sharpy/_sharpy")],
    cmdclass=dict(
        # Enable the CMakeExtension entries defined above
        build_ext=build_ext  # cmake_build_extension.BuildExtension,
    ),
)
