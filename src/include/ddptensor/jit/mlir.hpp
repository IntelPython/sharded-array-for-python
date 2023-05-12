// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ddptensor/CppTypes.hpp>
#include <ddptensor/tensor_i.hpp>

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

class Transceiver;

namespace jit {

template <typename T> struct PT_DTYPE {};
template <> struct PT_DTYPE<double> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::F64;
};
template <> struct PT_DTYPE<float> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::F32;
};
template <> struct PT_DTYPE<int64_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::I64;
};
template <> struct PT_DTYPE<int32_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::I32;
};
template <> struct PT_DTYPE<int16_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::I16;
};
template <> struct PT_DTYPE<int8_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::I8;
};
template <> struct PT_DTYPE<uint64_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::U64;
};
template <> struct PT_DTYPE<uint32_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::U32;
};
template <> struct PT_DTYPE<uint16_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::U16;
};
template <> struct PT_DTYPE<uint8_t> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::U8;
};
template <> struct PT_DTYPE<bool> {
  constexpr static ::imex::ptensor::DType value = ::imex::ptensor::I1;
};

inline ::imex::ptensor::DType getPTDType(DTypeId dt) {
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

// function type used for reporting back tensor results generated
// by Deferred::generate_mlir
using SetResFunc = std::function<void(
    Transceiver *transceiver, uint64_t rank, void *allocated, void *aligned,
    intptr_t offset, const intptr_t *sizes, const intptr_t *strides,
    int64_t *gs_allocated, int64_t *gs_aligned, uint64_t *lo_allocated,
    uint64_t *lo_aligned, uint64_t balanced)>;
using ReadyFunc = std::function<void(id_type guid)>;

// initialize jit
void init();

/// Manages iput/output (tensor) dependencies
class DepManager {
private:
  using IdValueMap = std::map<id_type, ::mlir::Value>;
  using IdCallbackMap = std::map<id_type, SetResFunc>;
  using IdReadyMap = std::map<id_type, std::vector<ReadyFunc>>;
  using IdRankMap = std::map<id_type, int>;
  // here we store the future (and not the guid) to keep it alive long enough
  using ArgList = std::vector<std::pair<id_type, tensor_i::future_type>>;

  ::mlir::func::FuncOp &_func; // MLIR function to which ops are added
  IdValueMap _ivm;             // guid -> mlir::Value
  IdCallbackMap _icm;          // guid -> deliver-callback
  IdReadyMap _icr;             // guid -> ready-callback
  IdRankMap _irm;              // guid -> rank as computed in MLIR
  ArgList _args;               // input arguments of the generated function

public:
  DepManager(::mlir::func::FuncOp &f) : _func(f) {}
  /// @return the ::mlir::Value representing the tensor with guid guid
  /// If the tensor is not created within the current function, it will
  /// be added as a function argument.
  ::mlir::Value getDependent(::mlir::OpBuilder &builder, id_type guid);

  /// for the tensor guid register the ::mlir::value and a callback to deliver
  /// the promise which generated the value if the tensor is alive when the
  /// function returns it will be added to the list of results
  void addVal(id_type guid, ::mlir::Value val, SetResFunc cb);
  void addReady(id_type guid, ReadyFunc cb);

  /// signals end of lifetime of given tensor: does not need to be returned
  void drop(id_type guid);

  /// create return statement and add results to function
  /// this must be called after store_inputs
  /// @return size of output in number of intptr_t's
  uint64_t handleResult(::mlir::OpBuilder &builder);

  /// devlier promise after execution
  void deliver(std::vector<intptr_t> &, uint64_t);

  /// store all inputs into given buffer
  /// This must be called before handleResults()
  std::vector<void *> store_inputs();
};

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
  ::mlir::MLIRContext _context;
  ::mlir::PassManager _pm;
  int _verbose;
  bool _useCache;
  ::mlir::SmallVector<::llvm::StringRef> _sharedLibPaths;
};

// size of memreftype in number of intptr_t's
inline uint64_t memref_sz(int rank) { return 3 + 2 * rank; }
inline uint64_t dtensor_sz(int rank) {
  return 2 * memref_sz(1) + memref_sz(rank) + 1;
};

} // namespace jit
