import cmake_build_extension
from setuptools import setup
from pathlib import Path

ext_modules = [
        cmake_build_extension.CMakeExtension(
            name="_ddptensor",
            # Name of the resulting package name (import mymath_pybind11)
            install_prefix="ddptensor",
            # Note: pybind11 is a build-system requirement specified in pyproject.toml,
            #       therefore pypa/pip or pypa/build will install it in the virtual
            #       environment created in /tmp during packaging.
            #       This cmake_depends_on option adds the pybind11 installation path
            #       to CMAKE_PREFIX_PATH so that the example finds the pybind11 targets
            #       even if it is not installed in the system.
            cmake_depends_on=["pybind11"],
            # Exposes the binary print_answer to the environment.
            # It requires also adding a new entry point in setup.cfg.
            # expose_binaries=["bin/print_answer"],
            # Writes the content to the top-level __init__.py
            #write_top_level_init=init_py,
            # Selects the folder where the main CMakeLists.txt is stored
            # (it could be a subfolder)
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
            ]
        ),
    ]

setup(name="ddptensor",
      version="0.1",
      description="Distributed Tensor and more",
      packages=["ddptensor", "ddptensor.numpy", "ddptensor.torch"],
      ext_modules=ext_modules,
      cmdclass=dict(
          # Enable the CMakeExtension entries defined above
          build_ext=cmake_build_extension.BuildExtension,
      ),
)
