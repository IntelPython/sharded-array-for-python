import os
import pathlib

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


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
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)
        # example of cmake args
        config = "Debug"  # if self.debug else 'RelWithDebInfo' #'Release'
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_BUILD_TYPE={config}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
            "-G=Ninja",
            "-DLLVM_ENABLE_LLD=ON",
            f"-DCMAKE_PREFIX_PATH={os.getenv('CONDA_PREFIX')}/lib/cmake",
        ]

        # example of build args
        build_args = ["--config", config]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "-j5"] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


setup(
    name="sharpy",
    version="0.2",
    description="Distributed array and more",
    packages=["sharpy", "sharpy.numpy"],  # "sharpy.torch"],
    ext_modules=[CMakeExtension("sharpy/_sharpy")],
    cmdclass=dict(
        # Enable the CMakeExtension entries defined above
        build_ext=build_ext  # cmake_build_extension.BuildExtension,
    ),
)
