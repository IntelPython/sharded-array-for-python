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

template <typename T> struct PT_DTYPE {};
template <> struct PT_DTYPE<double> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::F64;
};
template <> struct PT_DTYPE<float> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::F32;
};
template <> struct PT_DTYPE<int64_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::I64;
};
template <> struct PT_DTYPE<int32_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::I32;
};
template <> struct PT_DTYPE<int16_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::I16;
};
template <> struct PT_DTYPE<int8_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::I8;
};
template <> struct PT_DTYPE<uint64_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::U64;
};
template <> struct PT_DTYPE<uint32_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::U32;
};
template <> struct PT_DTYPE<uint16_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::U16;
};
template <> struct PT_DTYPE<uint8_t> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::U8;
};
template <> struct PT_DTYPE<bool> {
  constexpr static ::imex::ndarray::DType value = ::imex::ndarray::I1;
};

inline ::imex::ndarray::DType getPTDType(DTypeId dt) {
  switch (dt) {
  case FLOAT64:
    return PT_DTYPE<TYPE<FLOAT64>::dtype>::value;
  case INT64:
    return PT_DTYPE<TYPE<INT64>::dtype>::value;
  case FLOAT32:
    return PT_DTYPE<TYPE<FLOAT32>::dtype>::value;
  case INT32:
    return PT_DTYPE<TYPE<INT32>::dtype>::value;
  case INT16:
    return PT_DTYPE<TYPE<INT16>::dtype>::value;
  case INT8:
    return PT_DTYPE<TYPE<INT8>::dtype>::value;
  case UINT64:
    return PT_DTYPE<TYPE<UINT64>::dtype>::value;
  case UINT32:
    return PT_DTYPE<TYPE<UINT32>::dtype>::value;
  case UINT16:
    return PT_DTYPE<TYPE<UINT16>::dtype>::value;
  case UINT8:
    return PT_DTYPE<TYPE<UINT8>::dtype>::value;
  case BOOL:
    return PT_DTYPE<TYPE<BOOL>::dtype>::value;
  default:
    throw std::runtime_error("unknown dtype");
  }
}

// function type used for reporting back array results generated
// by Deferred::generate_mlir
using SetResFunc = std::function<void(
    uint64_t rank, void *l_allocated, void *l_aligned, intptr_t l_offset,
    const intptr_t *l_sizes, const intptr_t *l_strides, void *o_allocated,
    void *o_aligned, intptr_t o_offset, const intptr_t *o_sizes,
    const intptr_t *o_strides, void *r_allocated, void *r_aligned,
    intptr_t r_offset, const intptr_t *r_sizes, const intptr_t *r_strides,
    std::vector<int64_t> &&l_offs)>;
using ReadyFunc = std::function<void(id_type guid)>;

// initialize jit
void init();
void fini();

// A class to manage the MLIR business (compilation and execution).
// Just a stub for now, will need to be extended with paramters and maybe more.
class JIT {
public:
  JIT();
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
  std::string _crunnerlib, _runnerlib, _gpulib;
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
        : _guid(guid), _rank(rank), _value(value), _isDist(isDist),
          _setResFunc(setResFunc), _readyFuncs(readyFuncs) {}
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

::mlir::SmallVector<::mlir::Attribute> mkEnvs(::mlir::Builder &builder,
                                              int64_t rank,
                                              const std::string &device,
                                              uint64_t team);

} // namespace jit
} // namespace SHARPY
