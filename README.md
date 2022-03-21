[![.github/workflows/ci.yml](https://github.com/intel-sandbox/personal.fschlimb.ddptensor/actions/workflows/ci.yml/badge.svg)](https://github.com/intel-sandbox/personal.fschlimb.ddptensor/actions/workflows/ci.yml)
# Distributed Data-Parallel Python Tensor
A tensor implementation following the [array API as defined by the data-API consortium](https://data-apis.org/array-api/latest/index.html).
It supports a controller-worker execution model as well as a CSP-like execution.

## Setting up build environment
``` bash
git --recurse-submodules clone https://github.com/intel-sandbox/personal.fschlimb.ddptensor
cd personal.fschlimb.ddptensor
conda env create -f conda-env.yml -n ddpt
conda activate ddpt
export MPIROOT=$CONDA_PREFIX
export MKLROOT=$CONDA_PREFIX
```
## Building ddptensor
``` bash
CC=gcc-9 CXX=g++-9 CMAKE_BUILD_PARALLEL_LEVEL=8 python setup.py develop
```

## Running Tests
``` bash
# single rank
pytest test
# multiple ranks, controller-worker, controller spawns ranks
DDPT_MPI_SPAWN=$NoW PYTHON_EXE=`which python` pytest test
# multiple ranks, controller-worker, mpirun
mpirun -n $N python -m pytest test
# multiple ranks, CSP
DDPT_CW=0 mpirun -n $N python -m pytest test
```
