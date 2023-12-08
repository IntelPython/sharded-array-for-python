[![.github/workflows/ci.yml](https://github.com/intel-sandbox/personal.fschlimb.sharpy/actions/workflows/ci.yml/badge.svg)](https://github.com/intel-sandbox/personal.fschlimb.sharpy/actions/workflows/ci.yml)
# Distributed Data-Parallel Python Array
A array implementation following the [array API as defined by the data-API consortium](https://data-apis.org/array-api/latest/index.html).
It supports a controller-worker execution model as well as a CSP-like execution.

## Setting up build environment
Install MLIR/LLVM and IMEX from branch dist-ndarray (see https://github.com/intel-innersource/frameworks.ai.mlir.mlir-extensions/tree/dist-ndarray).
```bash
git --recurse-submodules clone https://github.com/intel-sandbox/personal.fschlimb.sharpy
cd personal.fschlimb.sharpy
git checkout jit
conda env create -f conda-env.yml -n sharpy
conda activate sharpy
export MPIROOT=$CONDA_PREFIX
export MLIRROOT=<your-MLIR-install-dir>
export IMEXROOT=<your-IMEX-install-dir>
```
## Building sharpy
```bash
python setup.py develop
```
If your compiler does not default to a recent version, try something like `CC=gcc-9 CXX=g++-9 python setup.py develop`

## Running Tests [non functional]
__Test are currently not operational on this branch.__

```bash
# single rank
pytest test
# multiple ranks, controller-worker, controller spawns ranks
SHARPY_MPI_SPAWN=$NoW PYTHON_EXE=`which python` pytest test
# multiple ranks, controller-worker, mpirun
mpirun -n $N python -m pytest test
# multiple ranks, CSP
SHARPY_CW=0 mpirun -n $N python -m pytest test
```

If SHARPY_MPI_SPAWN is set it spawns the provided number of MPI processes.
By default new processes launch python executing a worker loop.
This requires setting PYTHON_EXE.
Alternatively SHARPY_MPI_EXECUTABLE and SHARPY_MPI_EXE_ARGS are used.
Additionally SHARPY_MPI_HOSTS can be used to control the host to use for spawning processes.

## Running
```python
import sharpy as sp
sp.init(False)
a = sp.arange(0, 10, 1, sp.int64)
#print(a)       # should trigger compilation
b = sp.arange(0, 100, 10, sp.int64)
#print(b.dtype) # should _not_ trigger compilation
c = a * b
#print(c)
d = sp.sum(c, [0])
#del b          # generated function should _not_ return b
print(a, c, d) # printing of c (not a!) should trigger compilation
sp.fini()
```
Assuming the above is in file `simple.py` a single-process run is executed like
```bash
SHARPY_IDTR_SO=`pwd`/sharpy/libidtr.so python simple.py
```
and multi-process run is executed like
```bash
SHARPY_IDTR_SO=`pwd`/sharpy/libidtr.so mpirun -n 5 python simple.py
```

## Contributing
Please setup precommit hooks like this
```
pre-commit install -f -c ./.pre-commit-config.yaml
pre-commit autoupdate
```

## Overview
### Deferred Execution
Typically, sharpy operations do not get executed immediately. Instead, the function returns a transparent object (a future) only.
the actual computation gets deferred by creating a promise/deferred object and queuing it for later. This is not visible to users, they can use it as any other numpy-like library.

Only when actual data is needed, computation will happen; that is when
- the values of array elements are casted to bool int or float
- the array is printed

In the background a worker thread handles deferred objects. Until computation is needed it dequeues deferred objects from the FIFO queue and asks them to generate MLIR.
Objects can either generate MLIR or instead provide a run() function to immediately execute. For the latter case the current MLIR function gets executed before calling run() to make sure potential dependences are met.

### Distribution
Arrays and operations on them get transparently distributed across multiple processes. Respective functionality is partly handled by this library and partly IMEX dist dialect.
IMEX relies on a runtime library for complex communication tasks and for inspecting runtime configuration, such as number of processes and process id (MPI rank).
sharpy provides this library functionality in a separate dynamic library "idtr".

Right now, data is split in the first dimension (only). Each process knows the partition it owns. For optimization partitions can actually overlap.

sharpy supports to execution modes:
1. CSP/SPMD/explicitly-distributed execution, meaning all processes execute the same program, execution is replicated on all processes. Data is typically not replicated but distributed among processes.
2. Controller-Worker/implicitly-distributed execution, meaning only a single process executes the program and it distributes data and work to worker processes.
