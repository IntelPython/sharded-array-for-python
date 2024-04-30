[![CI](https://github.com/IntelPython/sharded-array-for-python/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/IntelPython/sharded-array-for-python/actions/workflows/ci.yml)

***This software package is not ready for production use and and merely a proof of concept implementation.***

# Sharded Array For Python

A array implementation following the [array API as defined by the data-API consortium](https://data-apis.org/array-api/latest/index.html).
Parallel and distributed execution currently is MPI/CSP-like. In a later version support for a controller-worker execution model will be added.

## Setting up build environment

Install MLIR/LLVM and Intel® Extension for MLIR (IMEX, see https://github.com/intel/mlir-extensions). Besides, Add "-DLLVM_ENABLE_RTTI=ON" into cmake arguments, and build target is "all" (not "check-imex").

```bash
git clone --recurse-submodules https://github.com/IntelPython/sharded-array-for-python
cd sharded-array-for-python
git checkout jit
conda create --file conda-env.txt --name sharpy
conda activate sharpy
export MPIROOT=$CONDA_PREFIX
export MLIRROOT=<your-MLIR-install-dir>
export IMEXROOT=<your-IMEX-install-dir>
```

## Building Sharded Array For Python

```bash
python -m pip install .
```

If your compiler does not default to a recent (e.g. g++ >= 9) version, try something like `CC=gcc-9 CXX=g++-9 python setup.py develop`

## Running Tests

```bash
# single rank
pytest test
# distributed on multiple ($N) ranks/processes
mpirun -n $N python -m pytest test
```

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
python simple.py
```

and multi-process run is executed like

```bash
mpirun -n 5 python simple.py
```

### Distributed Execution without mpirun

Instead of using mpirun to launch a set of ranks/processes, you can tell the runtime to
spawns ranks/processes for you by setting SHARPY_MPI_SPAWN to the number of desired MPI processes.
Additionally set SHARPY_MPI_EXECUTABLE and SHARPY_MPI_EXE_ARGS.
Additionally SHARPY_MPI_HOSTS can be used to control the host to use for spawning processes.

The following command will run the stencil example on 3 MPI ranks:

```bash
SHARPY_FALLBACK=numpy \
  SHARPY_MPI_SPAWN=2 \
  SHARPY_MPI_EXECUTABLE=`which python` \
  SHARPY_MPI_EXE_ARGS="examples/stencil-2d.py 10 2000 star 2" \
  python examples/stencil-2d.py 10 2000 star 2
```

## Contributing

Please setup pre-commit hooks like this

```bash
pre-commit install -f -c ./.pre-commit-config.yaml
pre-commit autoupdate
```

## Overview

### Deferred Execution

Typically, operations do not get executed immediately. Instead, the function returns a transparent object (a future) only.
The actual computation gets deferred by creating a promise/deferred object and queuing it for later. This is not visible to users, they can use it as any other numpy-like library.

Only when actual data is needed, computation will happen; that is when

- the values of array elements are casted to bool, int, float or string
- this includes when the array is printed

In the background a worker thread handles deferred objects. Until computation is needed it dequeues deferred objects from the FIFO queue and asks them to generate MLIR. Objects can either generate MLIR or instead provide a run() function to immediately execute. For the latter case the current MLIR function gets executed before calling run() to make sure potential dependencies are met.

### Distribution

Arrays and operations on them get transparently distributed across multiple processes. Respective functionality is partly handled by this library and partly IMEX dist dialect.
IMEX relies on a runtime library for complex communication tasks and for inspecting runtime configuration, such as number of processes and process id (MPI rank).
Sharded Array For Python provides this library functionality in a separate dynamic library "idtr".

Right now, data is split in the first dimension (only). Each process knows the partition it owns. For optimization partitions can actually overlap.

Sharded Array For Python currently supports one execution mode: CSP/SPMD/explicitly-distributed execution, meaning all processes execute the same program, execution is replicated on all processes. Data is typically not replicated but distributed among processes. The distribution is handled automatically, all operations on Sharded Arrays For Python can be viewed as collective operations.

Later, we'll add a Controller-Worker/implicitly-distributed execution mode, meaning only a single process executes the program and it distributes data and work to worker processes.

### Array API Coverage

Currently only a subset of the Array API is covered by Sharded Array For Python

- elementwise binary operations
- elementwise unary operations
- subviews (getitem with slices)
- assignment (setitem with slices)
- `empty`, `zeros`, `ones`, `linspace`, `arange`
- reduction operations over all dimensions (max, min, sum, ...)
- type promotion
- many cases of shape broadcasting

### Other Functionality

- `sharpy.to_numpy` converts a sharded array into a numpy array.
- `sharpy.numpy.from_function` allows creating a sharded array from a function (similar to numpy)
- In addition to the Array API Sharded Array For Python also provides functionality facilitating interacting with sharded arrays in a distributed environment.
  - `sharpy.spmd.gather` gathers the distributed array and forms a single, local and contiguous copy of the data as a numpy array
  - `sharpy.spmd.get_locals` return the local part of the distributed array as a numpy array
- sharpy allows providing a fallback array implementation. By setting SHARPY_FALLBACK to a python package it will call that package if a given function is not provided. It will pass sharded arrays as (gathered) numpy-arrays.

## Environment variables

### Compile time variables

Required to compile Sharded Array For Python:

- `MLIRROOT`: Set path to MLIR install root.
- `IMEXROOT`: Set path to Intel MLIR Extensions install root.

### Optional runtime variables

- `SHARPY_VERBOSE`: Set verbosity level. Accepted values 0-4, default 0.
- `SHARPY_FALLBACK`: Python package to call in case a function is not found in `sharpy`. For example, setting `SHARPY_FALLBACK=numpy` means that calling `sharpy.linalg.norm` would call `numpy.linalg.norm` for the entire (gathered) array.
- `SHARPY_PASSES`: Set MLIR pass pipeline. To see current pipeline run with `SHARPY_VERBOSE=1`.
- `SHARPY_FORCE_DIST`: Force code generation in distributed mode even if executed on a single process.
- `SHARPY_USE_CACHE`: Use in-memory JIT compile cache. Default 1.
- `SHARPY_OPT_LEVEL`: Set MLIR JIT compiler optimization level. Accepted values 0-3, default 3.
- `SHARPY_NO_ASYNC`: Do not use asynchronous MPI communication.
- `SHARPY_SKIP_COMM`: Skip all MPI communications. For debugging purposes only, can lead to incorrect results.

Device support:

- `SHARPY_DEVICE`: Set desided device, e.g., `"cpu"`, or `"gpu:0"`. By default CPU is used.
- `SHARPY_GPUX_SO`: Force path to GPU driver library file.

For [spawning MPI processes](#distributed-execution-without-mpirun):

- `SHARPY_MPI_SPAWN`: Number of MPI processes to spawn.
- `SHARPY_MPI_EXECUTABLE`: The executable to spawn.
- `SHARPY_MPI_EXE_ARGS`: Arguments to pass to the executable.
- `SHARPY_MPI_HOSTS`: Comma-separated list of hosts for MPI.
- `PYTHON_EXE`: Path to Python executable. Will be used if `SHARPY_MPI_EXECUTABLE` is undefined.
