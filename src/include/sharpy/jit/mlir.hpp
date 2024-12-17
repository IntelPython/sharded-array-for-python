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

mlir::Value shardNow(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     mlir::Value val, const std::string &team);

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
};

} // namespace jit
} // namespace SHARPY
