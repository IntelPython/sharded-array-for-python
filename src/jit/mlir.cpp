// SPDX-License-Identifier: BSD-3-Clause

/*
  Core MLIR functionality.
  - Adding/creating input and output to functions
  - Handling MLIR compiler machinery

  To reduce compile/link time only MLIR dialects and passes get
  registered/linked which are actually used.

  A typical jit cycle is controlled by the worker process executing
  process_promises.
  - create MLIR module/function
  - adding deferred operations
  - adding appropriate casts and return statements
  - updating function signature to accept existing arrays and returning new and
  live ones

  Typically operations have input dependencies, e.g. arrays produced by other
  operations. These can either come from outside the jit'ed function or be
  created within the function. Since we strictly add operations in serial order
  input dependencies must already exists. Deps are represented by guids and
  stored in the Registry.

  Internally sharpy's MLIR machinery keeps track of created and needed arrays.
  Those which were not created internally are added as input arguments to the
  jit-function. Those which are live (not destructed within the function) when
  the function is finalized are added as return values.

  MLIR/LLVM supports a single return value only. Following LLVM's policy we need
  to pack al return arrays into one large buffer/struct. Input arrays get
  represented as a series of arguments, as defined by MLIR/LLVM and IMEX's dist
  dialect.
*/

#include "sharpy/jit/mlir.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Registry.hpp"
#include "sharpy/UtilsAndTypes.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "llvm/Support/TargetSelect.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_sha1_ostream.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "sharpy/itac.hpp"

namespace SHARPY {
namespace jit {

std::string DepManager::_fname = "sharpy_jit";

DepManager::DepManager(jit::JIT &jit) : _jit(jit), _builder(&jit.context()) {
  auto loc = _builder.getUnknownLoc();

  // Create a MLIR module
  _module = _builder.create<::mlir::ModuleOp>(loc);
  // Create the jit func
  // create dummy type, we'll replace it with the actual type later
  auto dummyFuncType = _builder.getFunctionType({}, {});
  _func = _builder.create<::mlir::func::FuncOp>(loc, _fname, dummyFuncType);
  // create function entry block
  auto &entryBlock = *_func.addEntryBlock();
  // Set the insertion point in the _builder to the beginning of the function
  // body
  _builder.setInsertionPointToStart(&entryBlock);
}

void DepManager::finalizeAndRun() {
  // get input buffers (before results!)
  auto input = finalize_inputs();
  // create return statement and adjust function type
  uint64_t osz = handleResult(_builder);
  // also request generation of c-wrapper function
  _func->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                 _builder.getUnitAttr());
  if (_jit.verbose())
    _func.getFunctionType().dump();
  // add our only mesh
  if (getTransceiver()) {
    int64_t nRanks = (int64_t)getTransceiver()->nranks();
    auto mesh = getTransceiver()->mesh();
    ::mlir::OpBuilder::InsertionGuard g(_builder);
    _builder.setInsertionPointToStart(&_module.getRegion().front());
    auto meshOp = _builder.create<::mlir::mesh::MeshOp>(
        _builder.getUnknownLoc(), mesh, mlir::ArrayRef<int64_t>{nRanks});
    meshOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
  // add the function to the module
  _module.push_back(_func);

  if (osz > 0 || !input.empty()) {
    // compile and run the module
    auto output = _jit.run(_module, _fname, input, osz);
    // we assume we only store memrefdescriptors, e.g. arrays of inptr_t
    for (auto p : input) {
      delete[] reinterpret_cast<intptr_t *>(p);
    }
    if (output.size() != osz)
      throw std::runtime_error("failed running jit");

    // push results to deliver promises
    deliver(output, osz);
  } else {
    if (_jit.verbose())
      std::cerr << "\tskipping\n";
  }
}

DepManager::InOut *DepManager::findInOut(id_type guid) {
  for (auto &r : _inOut) {
    if (r._guid == guid) {
      return &r;
    }
  }
  return nullptr;
}

::mlir::Value DepManager::getDependent(::mlir::OpBuilder &builder,
                                       const array_i::future_type &fut) {
  id_type guid = fut.guid();
  if (auto d = findInOut(guid); !d) {
    // this must be an argument, so the future should be ready
    auto impl = std::dynamic_pointer_cast<NDArray>(fut.get());
    return addDependent(builder, impl.get());
  } else {
    return d->_value;
  }
};

static ::mlir::RankedTensorType getTensorType(size_t ndims, intptr_t /*offset*/,
                                              intptr_t *sizes,
                                              intptr_t * /*strides*/,
                                              ::mlir::Type elType) {
  return mlir::RankedTensorType::get(::mlir::ArrayRef(sizes, ndims), elType);
}

static ::mlir::RankedTensorType getTensorType(size_t ndims, const DynMemRef &mr,
                                              ::mlir::Type elType) {
  return getTensorType(ndims, mr._offset, mr._sizes, mr._strides, elType);
}

// Add a new array input argument to the jit function.
// The array is passed as memrefs.
// If array is distributed a sharding annotaion is added.
// Returns the value to be used by users of the array
::mlir::Value DepManager::addDependent(::mlir::OpBuilder &builder,
                                       const NDArray *impl) {
  id_type guid = impl->guid();
  // this must not yet be an input argument to the jit function
  if (findInOut(guid)) {
    throw std::runtime_error("Internal error: array already added");
  }

  auto idx = _lastIn;
  size_t ndims = impl->ndims();
  ::mlir::SmallVector<int64_t> zeros(ndims, 0);
  auto elType(getMLIRType(builder, impl->dtype()));
  auto loc = builder.getUnknownLoc();

  ::mlir::OpBuilder::InsertionGuard g(_builder);
  _builder.setInsertionPointToStart(&_func.front());

  auto storeMR = [ndims](const DynMemRef &mr) -> intptr_t * {
    intptr_t *buff = new intptr_t[memref_sz(ndims)];
    buff[0] = reinterpret_cast<intptr_t>(mr._allocated);
    buff[1] = reinterpret_cast<intptr_t>(mr._aligned);
    buff[2] = static_cast<intptr_t>(mr._offset);
    memcpy(buff + 3, mr._sizes, ndims * sizeof(intptr_t));
    memcpy(buff + 3 + ndims, mr._strides, ndims * sizeof(intptr_t));
    return buff;
  }; // FIXME memory leak?

  auto typ = getTensorType(ndims, impl->owned_data(), elType);
  _func.insertArgument(idx, typ, {}, loc);
  _inputs.push_back(storeMR(impl->owned_data()));
  ::mlir::Value arg = _func.getArgument(idx);
  _inOut.emplace_back(InOut(guid, arg));
  _lastIn += 1;

  if (!impl->team().empty() && ndims == 0) {
    // FIXME add sharding
  }

  return arg;
}

std::vector<void *> DepManager::finalize_inputs() {
  // return current buffer and reset internal input buffer
  _lastIn = 0;
  return std::move(_inputs);
}

void DepManager::addVal(id_type guid, ::mlir::Value val, SetResFunc cb) {
  if (findInOut(guid)) {
    throw std::runtime_error("Internal error: array already added");
  }
  auto tmp = _inOut.emplace_back(InOut(guid, val, cb));
}

void DepManager::addReady(id_type guid, ReadyFunc cb) {
  auto x = findInOut(guid);
  if (!x) {
    x = &_inOut.emplace_back(InOut{guid});
  }
  x->_readyFuncs.emplace_back(cb);
}

void DepManager::drop(id_type guid) {
  auto x = findInOut(guid);
  // drop from our list if it has not yet been delivered
  if (x) {
    x->_value = nullptr;
    x->_setResFunc = nullptr;
  }
}

// Now we have to define the return type as a ValueRange of all arrays which we
// have created (runnables have put them into DepManager when generating mlir)
// We also compute the total size of the struct llvm created for this return
// type llvm will basically return a struct with all the arrays as members, each
// of type JIT::MemRefDescriptor
uint64_t DepManager::handleResult(::mlir::OpBuilder &builder) {
  // Need a container to put all return values, will be used to construct
  // TypeRange
  mlir::OpBuilder::InsertionGuard guard(builder);
  std::vector<::mlir::Value> ret_values;

  // here we store the total size of the llvm struct
  auto loc = builder.getUnknownLoc();
  uint64_t sz = 0;
  unsigned idx = 0;
  for (auto &x : _inOut) {
    ::mlir::Value value = x._value;
    if (value) {
      bool isDist = false; // FIXME
      auto rank =
          mlir::cast<::mlir::RankedTensorType>(value.getType()).getRank();

      ret_values.emplace_back(value);
      _func.insertResult(idx, value.getType(), {});
      x._rank = rank;
      x._isDist = isDist;
      // add sizes of array
      sz += ndarray_sz(rank, isDist);
      ++idx;

      if (isDist && rank) {
        // FIXME return sharding/halo info
      }
    }
  }

  if (HAS_ITAC()) {
    int vtExeSym, vtSHARPYClass;
    VT(VT_classdef, "sharpy", &vtSHARPYClass);
    VT(VT_funcdef, "execute", vtSHARPYClass, &vtExeSym);
    ::mlir::Value s = builder.create<::mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(vtExeSym));
    auto end = builder.create<::mlir::func::CallOp>(
        builder.getUnknownLoc(), "VT_end",
        ::mlir::TypeRange(builder.getIntegerType(32)), ::mlir::ValueRange(s));
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(end->getBlock());
    (void)builder.create<::mlir::func::CallOp>(
        builder.getUnknownLoc(), "VT_begin",
        ::mlir::TypeRange(builder.getIntegerType(32)), ::mlir::ValueRange(s));
  }

  // add return statement
  (void)builder.create<::mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                               ret_values);

  return 2 * sz;
}

void DepManager::deliver(std::vector<intptr_t> &output, uint64_t sz) {
  size_t pos = 0;

  for (auto &x : _inOut) {
    if (x._value) {
      auto t_allocated = reinterpret_cast<void *>(output[pos]);
      auto t_aligned = reinterpret_cast<void *>(output[pos + 1]);
      intptr_t t_offset = output[pos + 2];
      intptr_t *t_sizes = &output[pos + 3];
      intptr_t *t_strides = &output[pos + 3 + x._rank];
      pos += memref_sz(x._rank);

      if (x._setResFunc) {
        x._setResFunc(x._rank, t_allocated, t_aligned, t_offset, t_sizes,
                      t_strides, // lData
                      {});
      }

      if (x._rank > 0 && x._isDist) {
        // FIXME upack sharding/halo info
      }
    }

    // not attached to value, call always if provided
    if (!x._readyFuncs.empty()) {
      for (auto cb : x._readyFuncs) {
        cb(x._guid);
      }
    }
  }

  // finally clear
  _inOut.clear();
}

static std::map<std::array<unsigned char, 20>,
                std::unique_ptr<::mlir::ExecutionEngine>>
    engineCache;

std::vector<intptr_t> JIT::run(::mlir::ModuleOp &module,
                               const std::string &fname,
                               std::vector<void *> &inp, size_t osz) {

  int vtSHARPYClass, vtHashSym, vtEEngineSym, vtRunSym, vtHashGenSym;
  if (HAS_ITAC()) {
    VT(VT_classdef, "sharpy", &vtSHARPYClass);
    VT(VT_funcdef, "lookup_cache", vtSHARPYClass, &vtHashSym);
    VT(VT_funcdef, "gen_sha", vtSHARPYClass, &vtHashGenSym);
    VT(VT_funcdef, "eengine", vtSHARPYClass, &vtEEngineSym);
    VT(VT_funcdef, "run", vtSHARPYClass, &vtRunSym);
    VT(VT_begin, vtEEngineSym);

    ::mlir::OpBuilder builder(module->getContext());
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    auto intTyp = builder.getIntegerType(32);
    auto funcType = builder.getFunctionType({intTyp}, {intTyp});
    builder.create<::mlir::func::FuncOp>(module.getLoc(), "VT_begin", funcType)
        .setPrivate();
    builder.create<::mlir::func::FuncOp>(module.getLoc(), "VT_end", funcType)
        .setPrivate();
  }

  ::mlir::ExecutionEngine *enginePtr;
  std::unique_ptr<::mlir::ExecutionEngine> tmpEngine;

  if (_useCache) {
    VT(VT_begin, vtHashGenSym);

    llvm::raw_sha1_ostream shaOS;
    module->print(shaOS);
    auto cksm = shaOS.sha1();
    VT(VT_end, vtHashGenSym);

    VT(VT_begin, vtHashSym);
    if (auto search = engineCache.find(cksm); search == engineCache.end()) {
      engineCache[cksm] = createExecutionEngine(module);
    } else {
      if (_verbose) {
        std::cerr << "cached..." << std::endl;
      }
    }
    enginePtr = engineCache[cksm].get();
    VT(VT_end, vtHashSym);
  } else {
    VT(VT_begin, vtHashSym);
    tmpEngine = createExecutionEngine(module);
    enginePtr = tmpEngine.get();
    VT(VT_end, vtHashSym);
  }

  auto expectedFPtr =
      enginePtr->lookupPacked(std::string("_mlir_ciface_") + fname);
  if (auto err = expectedFPtr.takeError()) {
    ::llvm::errs() << "JIT invocation failed: " << toString(std::move(err))
                   << "\n";
    throw std::runtime_error("JIT invocation failed");
  }
  auto jittedFuncPtr = *expectedFPtr;

  // pack function arguments
  llvm::SmallVector<void *> args;
  std::vector<intptr_t> out(osz);
  auto tmp = out.data();
  // first arg must be the result ptr
  if (osz) {
    args.push_back(&tmp);
  }
  // we need a void*& for every input array
  // we refer directly to the storage in inp
  for (auto &arg : inp) {
    args.push_back(&arg);
  }

  // call function
  (*jittedFuncPtr)(args.data());

  VT(VT_end, vtEEngineSym);
  return out;
}

std::unique_ptr<::mlir::ExecutionEngine>
JIT::createExecutionEngine(::mlir::ModuleOp &module) {
  if (_verbose)
    std::cerr << "compiling..." << std::endl;
  if (_verbose > 1)
    module.dump();

  // Create an ::mlir execution engine. The execution engine eagerly
  // JIT-compiles the module.
  ::mlir::ExecutionEngineOptions opts;
  opts.transformer = _optPipeline;
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::getLevel(_jit_opt_level);
  opts.sharedLibPaths = _sharedLibPaths;
  opts.enableObjectDump = true;

  // lower to LLVM
  if (::mlir::failed(_pm.run(module)))
    throw std::runtime_error("failed to run pass manager");

  if (_verbose > 2)
    module.dump();

  auto maybeEngine = ::mlir::ExecutionEngine::create(module, opts);
  if (!maybeEngine) {
    throw std::runtime_error("failed to construct an execution engine");
  }
  return std::move(maybeEngine.get());
}

static const std::string cpu_pipeline =
    "func.func(sharding-propagation),"
    "coalesce-shard-ops,"
    "canonicalize,"
    "func.func(mesh-spmdization),"
    "canonicalize,"
    "convert-mesh-to-mpi,"
    "canonicalize,"
    "convert-ndarray-to-linalg,"
    "func.func(tosa-to-linalg),"
    "func.func(tosa-to-tensor),"
    "linalg-generalize-named-ops,"
    "linalg-fuse-elementwise-ops,"
    "arith-expand,"
    "memref-expand,"
    "empty-tensor-to-alloc-tensor,"
    "canonicalize,"
    "one-shot-bufferize{bufferize-function-boundaries=1},"
    "canonicalize,"
    "imex-remove-temporaries,"
    "buffer-deallocation-pipeline,"
    "convert-bufferization-to-memref,"
    "func.func(convert-linalg-to-parallel-loops),"
    "func.func(scf-parallel-loop-fusion),"
    "drop-regions,"
    "canonicalize,"
    "fold-memref-alias-ops,"
    "expand-strided-metadata,"
    "convert-math-to-funcs,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "symbol-dce,"
    "finalize-memref-to-llvm,"
    "convert-math-to-llvm,"
    "convert-math-to-libm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts";

static const std::string gpu_pipeline =
    "add-gpu-regions,"
    "canonicalize,"
    "func.func(sharding-propagation),"
    "coalesce-shard-ops,"
    "canonicalize,"
    "func.func(mesh-spmdization),"
    "canonicalize,"
    "convert-mesh-to-mpi,"
    "canonicalize,"
    "convert-ndarray-to-linalg,"
    "func.func(tosa-to-linalg),"
    "func.func(tosa-to-tensor),"
    "linalg-generalize-named-ops,"
    "linalg-fuse-elementwise-ops,"
    "arith-expand,"
    "memref-expand,"
    "arith-bufferize,"
    "func-bufferize,"
    "func.func(empty-tensor-to-alloc-tensor),"
    "func.func(scf-bufferize),"
    "func.func(tensor-bufferize),"
    "func.func(bufferization-bufferize),"
    "func.func(linalg-bufferize),"
    "func.func(linalg-detensorize),"
    "func.func(tensor-bufferize),"
    "region-bufferize,"
    "canonicalize,"
    "func.func(finalizing-bufferize),"
    "imex-remove-temporaries,"
    "func.func(convert-linalg-to-parallel-loops),"
    "func.func(scf-parallel-loop-fusion),"
    // GPU
    "func.func(imex-add-outer-parallel-loop),"
    "func.func(gpu-map-parallel-loops),"
    "func.func(convert-parallel-loops-to-gpu),"
    // insert-gpu-allocs pass can have client-api = opencl or vulkan args
    "func.func(insert-gpu-allocs{in-regions=1}),"
    "drop-regions,"
    "canonicalize,"
    // "normalize-memrefs,"
    // "gpu-decompose-memrefs,"
    "func.func(lower-affine),"
    "gpu-kernel-outlining,"
    "canonicalize,"
    "cse,"
    // The following set-spirv-* passes can have client-api = opencl or vulkan
    // args
    "set-spirv-capabilities{client-api=opencl},"
    "gpu.module(set-spirv-abi-attrs{client-api=opencl}),"
    "canonicalize,"
    "fold-memref-alias-ops,"
    "imex-convert-gpu-to-spirv{enable-vc-intrinsic=1},"
    "spirv.module(spirv-lower-abi-attrs),"
    "spirv.module(spirv-update-vce),"
    // "func.func(llvm-request-c-wrappers),"
    "serialize-spirv,"
    "expand-strided-metadata,"
    "lower-affine,"
    "convert-gpu-to-gpux,"
    "convert-func-to-llvm,"
    "convert-math-to-llvm,"
    "convert-gpux-to-llvm,"
    "finalize-memref-to-llvm,"
    "reconcile-unrealized-casts";

const std::string _passes(get_text_env("SHARPY_PASSES"));
static const std::string &pass_pipeline =
    _passes != "" ? _passes : (useGPU() ? gpu_pipeline : cpu_pipeline);

JIT::JIT(const std::string &libidtr)
    : _context(::mlir::MLIRContext::Threading::DISABLED), _pm(&_context),
      _verbose(0), _jit_opt_level(3), _idtrlib(libidtr) {
  // Register the translation from ::mlir to LLVM IR, which must happen before
  // we can JIT-compile.
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::mlir::registerAllExtensions(registry);
  ::imex::registerAllDialects(registry);
  ::mlir::registerAllToLLVMIRTranslations(registry);
  _context.appendDialectRegistry(registry);

  // load the dialects we use
  _context.getOrLoadDialect<::imex::ndarray::NDArrayDialect>();
  _context.getOrLoadDialect<::imex::distruntime::DistRuntimeDialect>();
  _context.getOrLoadDialect<::imex::region::RegionDialect>();
  _context.getOrLoadDialect<::mlir::arith::ArithDialect>();
  _context.getOrLoadDialect<::mlir::func::FuncDialect>();
  _context.getOrLoadDialect<::mlir::linalg::LinalgDialect>();
  _context.getOrLoadDialect<::mlir::mesh::MeshDialect>();
  _context.getOrLoadDialect<::mlir::tosa::TosaDialect>();
  // create the pass pipeline from string
  if (::mlir::failed(::mlir::parsePassPipeline(pass_pipeline, _pm)))
    throw std::runtime_error("failed to parse pass pipeline");

  _verbose = get_int_env("SHARPY_VERBOSE");
  // some verbosity
  if (_verbose) {
    std::cerr << "SHARPY_PASSES=\"" << pass_pipeline << "\"" << std::endl;
    // _pm.enableStatistics();
    if (_verbose > 2)
      _pm.enableTiming();
    // if(_verbose > 1)
    //   _pm.dump();
    if (_verbose > 3)
      _pm.enableIRPrinting();
  }

  _useCache = get_bool_env("SHARPY_USE_CACHE", 1);
  std::cerr << "enableObjectDump=" << _useCache << std::endl;
  _jit_opt_level = get_int_env("SHARPY_OPT_LEVEL", 3);
  if (_jit_opt_level < 0 || _jit_opt_level > 3) {
    throw std::runtime_error(std::string("Invalid SHARPY_OPT_LEVEL"));
  }

  auto mlirRoot(get_text_env("MLIRROOT"));
  mlirRoot = !mlirRoot.empty() ? mlirRoot : std::string(CMAKE_MLIR_ROOT);
  _crunnerlib = mlirRoot + "/lib/libmlir_c_runner_utils.so";
  _runnerlib = mlirRoot + "/lib/libmlir_runner_utils.so";
  if (!std::ifstream(_crunnerlib)) {
    throw std::runtime_error("Cannot find libmlir_c_runner_utils.so");
  }
  if (!std::ifstream(_runnerlib)) {
    throw std::runtime_error("Cannot find libmlir_runner_utils.so");
  }

  if (useGPU()) {
    auto gpuxlibstr = get_text_env("SHARPY_GPUX_SO");
    if (!gpuxlibstr.empty()) {
      _gpulib = std::string(gpuxlibstr);
    } else {
      auto imexRoot = get_text_env("IMEXROOT");
      imexRoot = !imexRoot.empty() ? imexRoot : std::string(CMAKE_IMEX_ROOT);
      _gpulib = imexRoot + "/lib/liblevel-zero-runtime.so";
      if (!std::ifstream(_gpulib)) {
        throw std::runtime_error("Cannot find liblevel-zero-runtime.so");
      }
    }
    _sharedLibPaths = {_crunnerlib.c_str(), _runnerlib.c_str(),
                       _idtrlib.c_str(), _gpulib.c_str()};
  } else {
    _sharedLibPaths = {_crunnerlib.c_str(), _runnerlib.c_str(),
                       _idtrlib.c_str()};
  }

  // detect target architecture
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    throw std::runtime_error(
        "Failed to create a JITTargetMachineBuilder for the host\n");
  }

  // build TargetMachine
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    throw std::runtime_error("Failed to create a TargetMachine for the host\n");
  }
  _tm = std::move(tmOrError.get());

  // build optimizing pipeline
  _optPipeline = ::mlir::makeOptimizingTransformer(
      /*optLevel=*/_jit_opt_level,
      /*sizeLevel=*/0,
      /*targetMachine=*/_tm.get());
}

// register dialects and passes
void init() {
  static_assert(sizeof(intptr_t) == sizeof(void *));
  static_assert(sizeof(intptr_t) == sizeof(int64_t));
  if (!isText(pass_pipeline)) {
    throw std::runtime_error("Invalid SHARPY_PASSES");
  }

  ::mlir::registerAllPasses();
  ::imex::registerAllPasses();

  // Initialize LLVM targets.
  // llvm::InitLLVM y(0, nullptr);
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();
  ::llvm::InitializeNativeTargetAsmParser();
}

void fini() { engineCache.clear(); }
} // namespace jit
} // namespace SHARPY
