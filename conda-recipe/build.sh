# FIX dpcpp env
set -ex
if [ -n "${GCC}" ]; then
    export CC=${GCC}
fi
if [ -n "${GXX}" ]; then
    export CXX=${GXX}
fi
env
conda list

if [ -z ${GITHUB_WORKSPACE} ]; then
    INSTALLED_DIR=${SRC_DIR}/installed
else
    INSTALLED_DIR=$GITHUB_WORKSPACE/third_party/install
fi
cd ${SRC_DIR}

if [ ! -d "level-zero" ]; then
    git clone https://github.com/oneapi-src/level-zero.git
fi
pushd level-zero
git checkout v1.8.1
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${INSTALLED_DIR}/level-zero
cmake --build build --target install
popd

# skip checkout/build of IMEX/LLVM if there is an install
if [ ! -d "${INSTALLED_DIR}/imex/lib" ]; then
    export SYCL_DIR=${CONDA_PREFIX}
    echo "Using SYCL_DIR=${SYCL_DIR}"

    # grrrr
    # as long as the conda packages differ from system oneAPI installs we need to explicitly provide the include dir for SYCL
    # see also below in cmake call
    for root in "${CONDA_PREFIX}" "${BUILD_PREFIX}"; do
        spirvdir=$(find "${root}" -type d -name __spirv | head -n 1)
        if [ -d "${spirvdir}" ]; then
            SPIRV_INC_DIR=$(dirname "${spirvdir}")
            break
        fi
    done
    if [ -d "${SPIRV_INC_DIR}" ]; then
        echo "Using SPIRV_INC_DIR=${SPIRV_INC_DIR}"
        mkdir "${SRC_DIR}/grrrr/include"
        ln -s "${SPIRV_INC_DIR}" "${SRC_DIR}/grrrr/include/CL"
        SPIRV_INC_DIR="${SRC_DIR}/grrrr/include"
    else
        echo "Fatal error: SPIRV_INC_DIR not found"
        exit 1
    fi

    rm -rf ${INSTALLED_DIR}/imex
    IMEX_SHA=$(cat imex_version.txt)
    if [ ! -d "mlir-extensions" ]; then
        git clone --recurse-submodules --branch main --single-branch https://github.com/intel/mlir-extensions
    fi
    pushd mlir-extensions
    git reset --hard HEAD
    git fetch --prune
    git checkout $IMEX_SHA
    git apply ${RECIPE_DIR}/imex_*.patch
    LLVM_SHA=$(cat build_tools/llvm_version.txt)
    # if [ ! -d "llvm-project" ]; then ln -s ~/github/llvm-project .; fi
    if [ ! -d "llvm-project" ]; then
        mkdir llvm-project
        pushd llvm-project
        git init
        git remote add origin https://github.com/llvm/llvm-project
        git fetch origin ${LLVM_SHA}
        git reset --hard FETCH_HEAD
    else
        pushd llvm-project
        git reset --hard
        git checkout ${LLVM_SHA}
    fi
    if [ -d "${SRC_DIR}/mlir-extensions/build_tools/patches" ]; then
        git apply ${SRC_DIR}/mlir-extensions/build_tools/patches/*.patch
    fi
    popd
    rm -rf build/CMakeFiles/ build/CMakeCache.txt
    cmake -S ./llvm-project/llvm -B build \
        -GNinja \
        -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/cmake \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DCMAKE_INSTALL_PREFIX="${INSTALLED_DIR}/imex" \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_EXTERNAL_PROJECTS="Imex"  \
        -DLLVM_EXTERNAL_IMEX_SOURCE_DIR=. \
        -DLEVEL_ZERO_DIR=${INSTALLED_DIR}/level-zero \
        -DIMEX_ENABLE_SYCL_RUNTIME=1 \
        -DIMEX_ENABLE_L0_RUNTIME=1 \
        -DCMAKE_CXX_FLAGS="-I${SPIRV_INC_DIR}"  # grrrr
    cmake --build build
    cmake --install build --prefix=${INSTALLED_DIR}/imex
    popd
else
    echo "Found IMEX install, skipped building IMEX"
fi

ls -l ${CONDA_PREFIX}/lib
ZLIBROOT=${CONDA_PREFIX} MLIRROOT=${INSTALLED_DIR}/imex IMEXROOT=${INSTALLED_DIR}/imex ${PYTHON} -m pip install . -vv
