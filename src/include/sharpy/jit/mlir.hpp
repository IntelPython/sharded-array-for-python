// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <sharpy/CppTypes.hpp>
#include <sharpy/array_i.hpp>

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SHARPY {

class Transceiver;
class NDArray;

namespace jit {

inline ::mlir::Type getMLIRType(::mlir::OpBuilder &builder, DTypeId dt) {
  switch (dt) {
  case FLOAT64:
    return builder.getF64Type();
  case INT64:
  case UINT64:
    return builder.getI64Type();
  case FLOAT32:
    return builder.getF32Type();
  case INT32:
  case UINT32:
    return builder.getI32Type();
  case INT16:
  case UINT16:
    return builder.getI16Type();
  case INT8:
  case UINT8:
    return builder.getI8Type();
  case BOOL:
    return builder.getI1Type();
  default:
    throw std::invalid_argument("unknown dtype");
  }
}

// function type used for reporting back array results generated
// by Deferred::generate_mlir
using SetResFunc =
    std::function<void(uint64_t rank, void *allocated, void *aligned,
                       intptr_t offset, const intptr_t *sizes,
                       const intptr_t *strides, std::vector<int64_t> &&l_offs)>;
using ReadyFunc = std::function<void(id_type guid)>;

// initialize jit
void init();
void fini();

// A class to manage the MLIR business (compilation and execution).
// Just a stub for now, will need to be extended with parameters and maybe more.
class JIT {
public:
  JIT(const std::string &libidtr);
  // run
  std::vector<intptr_t> run(::mlir::ModuleOp &, const std::string &,
                            std::vector<void *> &, size_t);

  int verbose() { return _verbose; };
  ::mlir::MLIRContext &context() { return _context; };

private:
  std::unique_ptr<::mlir::ExecutionEngine>
  createExecutionEngine(::mlir::ModuleOp &module);
  ::mlir::MLIRContext _context;
  ::mlir::PassManager _pm;
  std::unique_ptr<llvm::TargetMachine> _tm;
  std::function<::llvm::Error(::llvm::Module *)> _optPipeline;
  int _verbose;
  bool _useCache;
  int _jit_opt_level;
  ::mlir::SmallVector<::llvm::StringRef> _sharedLibPaths;
  std::string _crunnerlib, _runnerlib, _idtrlib, _gpulib;
};

/// Manages input/output (array) dependencies
class DepManager {
private:
  struct InOut {
    id_type _guid = 0;
    ::mlir::Value _value = nullptr;
    SetResFunc _setResFunc;
    int _rank = 0;
    bool _isDist = false;
    std::vector<ReadyFunc> _readyFuncs;
    InOut(id_type guid = 0, const ::mlir::Value &value = nullptr,
          const SetResFunc &setResFunc = nullptr, int rank = 0,
          bool isDist = false, const std::vector<ReadyFunc> &readyFuncs = {})
        : _guid(guid), _value(value), _setResFunc(setResFunc), _rank(rank),
          _isDist(isDist), _readyFuncs(readyFuncs) {}
  };
  using InOutList = std::vector<InOut>;

  InOutList _inOut;
  int _lastIn = 0;
  JIT &_jit;
  ::mlir::OpBuilder _builder;
  ::mlir::ModuleOp _module;
  ::mlir::func::FuncOp _func;  // MLIR function to which ops are added
  std::vector<void *> _inputs; // collecting input args

  InOut *findInOut(id_type guid);
  static std::string _fname;

public:
  DepManager(JIT &jit);
  void finalizeAndRun();
  ::mlir::OpBuilder &getBuilder() { return _builder; }
  ::mlir::ModuleOp &getmodule() { return _module; }
  /// @return the ::mlir::Value representing the array
  /// If the array was not created within the current function, it will
  /// be added as a function argument.
  ::mlir::Value getDependent(::mlir::OpBuilder &builder,
                             const array_i::future_type &fut);
  ::mlir::Value addDependent(::mlir::OpBuilder &builder, const NDArray *fut);

  /// for the array guid register the ::mlir::value and a callback to deliver
  /// the promise which generated the value if the array is alive when the
  /// function returns it will be added to the list of results
  void addVal(id_type guid, ::mlir::Value val, SetResFunc cb);
  void addReady(id_type guid, ReadyFunc cb);

  /// signals end of lifetime of given array: does not need to be returned
  void drop(id_type guid);

  /// create return statement and add results to function
  /// this must be called after store_inputs
  /// @return size of output in number of intptr_t's
  uint64_t handleResult(::mlir::OpBuilder &builder);

  /// devlier promise after execution
  void deliver(std::vector<intptr_t> &, uint64_t);

  /// finalize all inputs storage and
  /// @return buffer with all inputs
  /// This must be called before handleResults()
  /// Resets internal input buffer
  std::vector<void *> finalize_inputs();
};

} // namespace jit
} // namespace SHARPY
