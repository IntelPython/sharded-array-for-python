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
  - updating function signature to accept existing tensors and returning new and
  live ones

  Typically operations have input dependences, e.g. tensors produced by other
  operations. These can either come from outside the jit'ed function or be
  created within the function. Since we strictly add operations in serial order
  input dependences must lready exists. Deps are represented by guids and stored
  in the Registry.

  Internally ddpt's MLIR machinery keeps track of created and needed tensors.
  Those which were not created internally are added as input arguments to the
  jit-function. Those which are live (not destructed within the function) when
  the function is finalized are added as return values.

  MLIR/LLVM supports a single return value only. Following LLVM's policy we need
  to pack al return tensors into one large buffer/struct. Input tensors get
  represented as a series of arguments, as defined by MLIR/LLVM and IMEX's dist
  dialect.
*/

#include "ddptensor/jit/mlir.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Registry.hpp"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
// #include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
// #include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
// #include "mlir/Dialect/GPU/Transforms/Passes.h"
// #include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
// #include "mlir/Dialect/NVGPU/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
// #include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
// #include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
// #include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
// #include "mlir/Dialect/Transform/Transforms/Passes.h"
// #include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
// #include <mlir/InitAllPasses.h>

#include "mlir/Parser/Parser.h"

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

#include <llvm/Support/raw_sha1_ostream.h>

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>

#include <cstdlib>
#include <iostream>
#include <string>

// #include "llvm/ADT/StringRef.h"
// #include "llvm/IR/Module.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorOr.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
// #include "llvm/Support/raw_ostream.h"

namespace jit {

static ::mlir::Type makeSignlessType(::mlir::Type type) {
  if (auto shaped = type.dyn_cast<::mlir::ShapedType>()) {
    auto origElemType = shaped.getElementType();
    auto signlessElemType = makeSignlessType(origElemType);
    return shaped.clone(signlessElemType);
  } else if (auto intType = type.dyn_cast<::mlir::IntegerType>()) {
    if (!intType.isSignless())
      return ::mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

// convert ddpt's DTYpeId into MLIR type
static ::mlir::Type getTType(::mlir::OpBuilder &builder, DTypeId dtype,
                             const ::mlir::SmallVector<int64_t> &gShape,
                             const ::mlir::SmallVector<int64_t> &lhShape,
                             const ::mlir::SmallVector<int64_t> &ownShape,
                             const ::mlir::SmallVector<int64_t> &rhShape,
                             uint64_t team, const uint64_t *lOffs) {
  ::mlir::Type etyp;

  switch (dtype) {
  case FLOAT64:
    etyp = builder.getF64Type();
    break;
  case FLOAT32:
    etyp = builder.getF32Type();
    break;
  case INT64:
  case UINT64:
    etyp = builder.getI64Type();
    break;
  case INT32:
  case UINT32:
    etyp = builder.getI32Type();
    break;
  case INT16:
  case UINT16:
    etyp = builder.getIntegerType(16);
    break;
  case INT8:
  case UINT8:
    etyp = builder.getI8Type();
    break;
  case BOOL:
    etyp = builder.getI1Type();
    break;
  default:
    throw std::runtime_error("unknown dtype");
  };

  if (team) {
    if (gShape.size()) {
      return ::imex::dist::DistTensorType::get(
          gShape, etyp, team,
          ::llvm::ArrayRef<int64_t>(reinterpret_cast<const int64_t *>(lOffs),
                                    gShape.size()),
          {lhShape, ownShape, rhShape});
    } else {
      auto eShp = ::mlir::SmallVector<int64_t>();
      return ::imex::dist::DistTensorType::get(eShp, etyp, team, {}, {eShp});
    }
  } else {
    return ::imex::ptensor::PTensorType::get(ownShape, etyp);
  }
}

::mlir::Value DepManager::getDependent(::mlir::OpBuilder &builder,
                                       id_type guid) {
  auto loc = builder.getUnknownLoc();
  if (auto d = _ivm.find(guid); d == _ivm.end()) {
    // Not found -> this must be an input argument to the jit function
    auto idx = _args.size();
    auto fut = Registry::get(guid);
    auto impl = std::dynamic_pointer_cast<DDPTensorImpl>(fut.get());
    auto rank = impl->ndims();
    ::mlir::SmallVector<int64_t> lhShape(rank), ownShape(rank), rhShape(rank);
    for (size_t i = 0; i < rank; i++) {
      lhShape[i] = impl->lh_shape() ? impl->lh_shape()[i] : 0;
      ownShape[i] = impl->local_shape()[i];
      rhShape[i] = impl->rh_shape() ? impl->rh_shape()[i] : 0;
    }
    auto typ = getTType(
        builder, impl->dtype(),
        ::mlir::SmallVector<int64_t>(impl->shape(), impl->shape() + rank),
        lhShape, ownShape, rhShape, fut.team(), impl->local_offsets());
    _func.insertArgument(idx, typ, {}, loc);
    auto val = _func.getArgument(idx);
    _args.push_back({guid, std::move(fut)});
    _ivm[guid] = val;
    return val;
  } else {
    return d->second;
  }
}

std::vector<void *> DepManager::store_inputs() {
  std::vector<void *> res;
  for (auto a : _args) {
    a.second.get().get()->add_to_args(res);
    _ivm.erase(a.first); // inputs need no delivery
    _icm.erase(a.first);
  }
  return res;
}

void DepManager::addVal(id_type guid, ::mlir::Value val, SetResFunc cb) {
  assert(_ivm.find(guid) == _ivm.end());
  _ivm[guid] = val;
  _icm[guid] = cb;
}

void DepManager::addReady(id_type guid, ReadyFunc cb) {
  _icr[guid].emplace_back(cb);
}

void DepManager::drop(id_type guid) {
  _ivm.erase(guid);
  _icm.erase(guid);
  _icr.erase(guid);
  Registry::del(guid);
  // FIXME create delete op
}

// Now we have to define the return type as a ValueRange of all arrays which we
// have created (runnables have put them into DepManager when generating mlir)
// We also compute the total size of the struct llvm created for this return
// type llvm will basically return a struct with all the arrays as members, each
// of type JIT::MemRefDescriptor
uint64_t DepManager::handleResult(::mlir::OpBuilder &builder) {
  // Need a container to put all return values, will be used to construct
  // TypeRange
  std::vector<::mlir::Value> ret_values(_ivm.size());

  // remove default result
  //_func.eraseResult(0);

  // here we store the total size of the llvm struct
  auto loc = builder.getUnknownLoc();
  uint64_t sz = 0;
  unsigned idx = 0;
  for (auto &v : _ivm) {
    ::mlir::Value value = v.second;
    // append the type and array/value
    if (!value.getType().isa<::imex::dist::DistTensorType>()) {
      value = builder.create<::imex::dist::CastOp>(loc, value);
    }
    ret_values[idx] = value;
    _func.insertResult(idx, value.getType(), {});
    auto rank = value.getType().cast<::imex::dist::DistTensorType>().getRank();
    _irm[v.first] = rank;
    // add sizes of dtensor (3 memrefs + team + balanced) to sz
    sz += dtensor_sz(rank);
    // clear reference to MLIR value
    v.second = nullptr;
    ++idx;
  }

  // add return statement
  auto ret_value = builder.create<::mlir::func::ReturnOp>(
      builder.getUnknownLoc(), ret_values);

  // _ivm defines the order of return values -> do not clear

  return 2 * sz;
}

void DepManager::deliver(std::vector<intptr_t> &outputV, uint64_t sz) {
  auto output = outputV.data();
  size_t pos = 0;

  auto getMR = [](int rank, auto buff, void *&allocated, void *&aligned,
                  intptr_t &offset, intptr_t *&sizes, intptr_t *&strides) {
    allocated = reinterpret_cast<void *>(buff[0]);
    aligned = reinterpret_cast<void *>(buff[1]);
    offset = buff[2];
    sizes = &buff[3];
    strides = &buff[3 + rank];
    return memref_sz(rank);
  };

  // _ivm defines the order of return values
  for (auto &r : _ivm) {
    auto guid = r.first;
    if (auto v = _icm.find(guid); v != _icm.end()) {
      assert(v->first == guid);
      auto rank = _irm[guid];
      // first extract team
      // auto team = output[pos];
      // pos += 1;
      // then tensors
      void *t_allocated[3];
      void *t_aligned[3];
      intptr_t t_offset[3];
      intptr_t *t_sizes[3];
      intptr_t *t_strides[3];
      if (rank > 0) {
        for (auto t = 0; t < 3; ++t) {
          pos += getMR(rank, &output[pos], t_allocated[t], t_aligned[t],
                       t_offset[t], t_sizes[t], t_strides[t]);
        }
        // lastly extract local offsets
        uint64_t *lo_allocated = reinterpret_cast<uint64_t *>(output[pos]);
        uint64_t *lo_aligned = reinterpret_cast<uint64_t *>(output[pos + 1]);
        intptr_t lo_offset = output[pos + 2];
        // no sizes/stride needed, just skip
        pos += memref_sz(1);
        // call finalization callback
        v->second(
            rank, t_allocated[0], t_aligned[0], t_offset[0], t_sizes[0],
            t_strides[0], // lhsHalo
            t_allocated[1], t_aligned[1], t_offset[1], t_sizes[1],
            t_strides[1], // lData
            t_allocated[2], t_aligned[2], t_offset[2], t_sizes[2],
            t_strides[2], // rhsHalo
            lo_allocated,
            lo_aligned + lo_offset // local offset is 1d tensor of uint64_t
        );
      } else { // 0d tensor
        pos += getMR(rank, &output[pos], t_allocated[1], t_aligned[1],
                     t_offset[1], t_sizes[1], t_strides[1]);
        v->second(rank, nullptr, nullptr, 0, nullptr, nullptr, // lhsHalo
                  t_allocated[1], t_aligned[1], t_offset[1], t_sizes[1],
                  t_strides[1],                          // lData
                  nullptr, nullptr, 0, nullptr, nullptr, // lhsHalo
                  nullptr, nullptr);
      }
    } else {
      assert(false);
    }
  }

  // ready signals will always be sent, at this point they are not linked to a
  // return value
  for (auto &readyV : _icr) {
    for (auto cb : readyV.second) {
      cb(readyV.first);
    }
  }
}

std::vector<intptr_t> JIT::run(::mlir::ModuleOp &module,
                               const std::string &fname,
                               std::vector<void *> &inp, size_t osz) {

  ::mlir::ExecutionEngine *enginePtr;
  std::unique_ptr<::mlir::ExecutionEngine> tmpEngine;

  if (_useCache) {
    static std::map<std::array<unsigned char, 20>,
                    std::unique_ptr<::mlir::ExecutionEngine>>
        engineCache;

    llvm::raw_sha1_ostream shaOS;
    module->print(shaOS);
    auto cksm = shaOS.sha1();

    if (auto search = engineCache.find(cksm); search == engineCache.end()) {
      engineCache[cksm] = createExecutionEngine(module);
    } else {
      if (_verbose)
        std::cerr << "cached..." << std::endl;
    }
    enginePtr = engineCache[cksm].get();
  } else {
    tmpEngine = createExecutionEngine(module);
    enginePtr = tmpEngine.get();
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
  // we need a void*& for every input tensor
  // we refer directly to the storage in inp
  for (auto &arg : inp) {
    args.push_back(&arg);
  }

  // call function
  (*jittedFuncPtr)(args.data());

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
  assert(maybeEngine && "failed to construct an execution engine");
  return std::move(maybeEngine.get());
}

static const char *pass_pipeline =
    getenv("DDPT_PASSES") ? getenv("DDPT_PASSES")
                          : "func.func(ptensor-dist),"
                            "func.func(dist-coalesce),"
                            "func.func(dist-infer-elementwise-cores),"
                            "convert-dist-to-standard,"
                            "canonicalize,"
                            "overlap-comm-and-compute,"
                            "lower-distruntime-to-idtr,"
                            "convert-ptensor-to-linalg,"
                            "canonicalize,"
                            "func.func(tosa-to-linalg),"
                            "func.func(tosa-to-tensor),"
                            "canonicalize,"
                            "linalg-fuse-elementwise-ops,"
                            // "convert-shape-to-std,"
                            "arith-expand,"
                            "memref-expand,"
                            "arith-bufferize,"
                            // "func-bufferize,"
                            "func.func(empty-tensor-to-alloc-tensor),"
                            "func.func(scf-bufferize),"
                            "func.func(tensor-bufferize),"
                            "func.func(bufferization-bufferize),"
                            "func.func(linalg-bufferize),"
                            "func.func(linalg-detensorize),"
                            "func.func(tensor-bufferize),"
                            "func.func(finalizing-bufferize),"
                            "func.func(buffer-deallocation),"
                            "imex-remove-temporaries,"
                            "func.func(convert-linalg-to-parallel-loops),"
                            "func.func(scf-parallel-loop-fusion),"
                            "canonicalize,"
                            "fold-memref-alias-ops,"
                            "expand-strided-metadata,"
                            "convert-math-to-funcs,"
                            "lower-affine,"
                            "convert-scf-to-cf,"
                            "finalize-memref-to-llvm,"
                            "convert-math-to-llvm,"
                            "convert-math-to-libm,"
                            "convert-func-to-llvm,"
                            "reconcile-unrealized-casts";
JIT::JIT()
    : _context(::mlir::MLIRContext::Threading::DISABLED), _pm(&_context),
      _verbose(0), _jit_opt_level(3) {
  // Register the translation from ::mlir to LLVM IR, which must happen before
  // we can JIT-compile.
  ::mlir::registerLLVMDialectTranslation(_context);
  ::mlir::registerBuiltinDialectTranslation(_context);
  ::mlir::registerOpenMPDialectTranslation(_context);
  // load the dialects we use
  _context.getOrLoadDialect<::mlir::arith::ArithDialect>();
  _context.getOrLoadDialect<::mlir::func::FuncDialect>();
  _context.getOrLoadDialect<::mlir::linalg::LinalgDialect>();
  _context.getOrLoadDialect<::imex::ptensor::PTensorDialect>();
  _context.getOrLoadDialect<::imex::dist::DistDialect>();
  _context.getOrLoadDialect<::imex::distruntime::DistRuntimeDialect>();
  // create the pass pipeline from string
  if (::mlir::failed(::mlir::parsePassPipeline(pass_pipeline, _pm)))
    throw std::runtime_error("failed to parse pass pipeline");

  const char *v_ = getenv("DDPT_VERBOSE");
  if (v_) {
    _verbose = std::stoi(v_);
  }
  // some verbosity
  if (_verbose) {
    std::cerr << "DDPT_PASSES=\"" << pass_pipeline << "\"" << std::endl;
    // _pm.enableStatistics();
    if (_verbose > 2)
      _pm.enableTiming();
    // if(_verbose > 1)
    //   _pm.dump();
    if (_verbose > 3)
      _pm.enableIRPrinting();
  }

  const char *envptr = getenv("DDPT_USE_CACHE");
  envptr = envptr ? envptr : "1";
  {
    auto c = std::string(envptr);
    _useCache = c == "1" || c == "y" || c == "Y" || c == "on" || c == "ON";
    std::cerr << "enableObjectDump=" << _useCache << std::endl;
  }
  const char *ol_ = getenv("DDPT_OPT_LEVEL");
  if (ol_) {
    _jit_opt_level = std::stoi(ol_);
    if (_jit_opt_level < 0 || _jit_opt_level > 3) {
      throw std::runtime_error(std::string("Bad optimization level: ") + ol_);
    }
  }

  const char *mlirRoot = getenv("MLIRROOT");
  mlirRoot = mlirRoot ? mlirRoot : CMAKE_MLIR_ROOT;
  _crunnerlib = std::string(mlirRoot) + "/lib/libmlir_c_runner_utils.so";
  _runnerlib = std::string(mlirRoot) + "/lib/libmlir_runner_utils.so";
  const char *idtrlib = getenv("DDPT_IDTR_SO");
  idtrlib = idtrlib ? idtrlib : "libidtr.so";
  _sharedLibPaths = {idtrlib, _crunnerlib.c_str(), _runnerlib.c_str()};

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
// adding everything leads to even more excessive compile/link time.
void init() {
  assert(sizeof(intptr_t) == sizeof(void *));
  assert(sizeof(intptr_t) == sizeof(uint64_t));
  // ::mlir::registerAllPasses();
  ::mlir::registerSCFPasses();
  ::mlir::registerSCFToControlFlowPass();
  ::mlir::registerConvertSCFToOpenMPPass();
  ::mlir::registerShapePasses();
  ::mlir::registerConvertShapeToStandardPass();
  ::mlir::tensor::registerTensorPasses();
  ::mlir::registerLinalgPasses();
  ::mlir::registerTosaToLinalg();
  ::mlir::registerTosaToTensor();
  ::mlir::registerConvertMathToFuncs();
  ::mlir::registerConvertMathToLibm();
  ::mlir::registerConvertMathToLLVMPass();
  ::mlir::tosa::registerTosaOptPasses();
  ::mlir::func::registerFuncPasses();
  ::mlir::registerConvertFuncToLLVMPass();
  ::mlir::bufferization::registerBufferizationPasses();
  ::mlir::arith::registerArithPasses();
  ::mlir::registerCanonicalizerPass();
  ::mlir::registerConvertAffineToStandardPass();
  ::mlir::registerFinalizeMemRefToLLVMConversionPass();
  ::mlir::registerArithToLLVMConversionPass();
  ::mlir::registerConvertMathToLLVMPass();
  ::mlir::registerConvertControlFlowToLLVMPass();
  ::mlir::registerConvertOpenMPToLLVMPass();
  ::mlir::memref::registerMemRefPasses();
  ::mlir::registerReconcileUnrealizedCastsPass();

  ::imex::registerPTensorPasses();
  ::imex::registerDistPasses();
  ::imex::registerDistRuntimePasses();
  ::imex::registerConvertDistToStandard();
  ::imex::registerConvertPTensorToLinalg();

  ::imex::registerRemoveTemporariesPass();

  // ::mlir::DialectRegistry registry;
  // ::mlir::registerAllDialects(registry);
  // ::imex::registerAllDialects(registry);

  // Initialize LLVM targets.
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();
  //::llvm::initializeLLVMPasses();
}
} // namespace jit
