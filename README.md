[![.github/workflows/ci.yml](https://github.com/intel-sandbox/personal.fschlimb.ddptensor/actions/workflows/ci.yml/badge.svg)](https://github.com/intel-sandbox/personal.fschlimb.ddptensor/actions/workflows/ci.yml)
# Distributed Data-Parallel Python Tensor
A tensor implementation following the [array API as defined by the data-API consortium](https://data-apis.org/array-api/latest/index.html).
It supports a controller-worker execution model as well as a CSP-like execution.

## Setting up build environment
Install MLIR/LLVM and IMEX from branch refactor (see https://github.com/intel/mlir-extensions/tree/refactor).
```bash
git --recurse-submodules clone https://github.com/intel-sandbox/personal.fschlimb.ddptensor
cd personal.fschlimb.ddptensor
conda env create -f conda-env.yml -n ddpt
conda activate ddpt
export MPIROOT=$CONDA_PREFIX
export MKLROOT=$CONDA_PREFIX
export MLIRROOT=<your-MLIR-install-dir>
export IMEXROOT=<your-IMEX-install-dir>
```
## Building ddptensor
```bash
python setup.py develop
```
If your compiler does not default to a recent version, try something like `CC=gcc-9 CXX=g++-9 python setup.py develop`

## Running Tests
__Test are currently not operational on this branch.__

```bash
# single rank
pytest test
# multiple ranks, controller-worker, controller spawns ranks
DDPT_MPI_SPAWN=$NoW PYTHON_EXE=`which python` pytest test
# multiple ranks, controller-worker, mpirun
mpirun -n $N python -m pytest test
# multiple ranks, CSP
DDPT_CW=0 mpirun -n $N python -m pytest test
```

If DDPT_MPI_SPAWN is set it spawns the provided number of MPI processes.
By default new processes launch python executing a worker loop.
This requires setting PYTHON_EXE.
Alternatively DDPT_MPI_EXECUTABLE and DDPT_MPI_EXE_ARGS are used.
Additionally DDPT_MPI_HOSTS can be used to control the host to use for spawning processes.

## Running
```python
import ddptensor as dt
dt.init(False)
a = dt.arange(0, 10, 1, dt.int64)
#print(a)       # should trigger compilation
b = dt.arange(0, 100, 10, dt.int64)
#print(b.dtype) # should _not_ trigger compilation
c = a * b
#print(c)
d = dt.sum(c, [0])
#del b          # generated function should _not_ return b
print(a, c, d) # printing of c (not a!) should trigger compilation
dt.fini()
```
Assuming the above is in file `simple.py` a single-process run is executed like
```bash
python DDPT_IDTR_SO=`pwd`/ddptensor/libidtr.so python simple.py
```
and multi-process run is executed like
```bash
python DDPT_IDTR_SO=`pwd`/ddptensor/libidtr.so mpirun -n 5 python simple.py
```

## Contributing
Please setup precommit hooks like this
```
pre-commit install -f -c ./.pre-commit-config.yaml
pre-commit autoupdate
```
