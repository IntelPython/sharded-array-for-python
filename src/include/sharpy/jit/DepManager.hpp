// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <sharpy/CppTypes.hpp>
#include <sharpy/array_i.hpp>
#include <sharpy/jit/mlir.hpp>
#include <vector>

namespace SHARPY {
namespace jit {

// function type used for reporting back array results generated
// by Deferred::generate_mlir
using SetResFunc =
    std::function<void(uint64_t rank, void *allocated, void *aligned,
                       intptr_t offset, const intptr_t *sizes,
                       const intptr_t *strides, std::vector<int64_t> &&l_offs)>;
using ReadyFunc = std::function<void(id_type guid)>;

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
  ::mlir::func::FuncOp _func;
  std::vector<void *> _inputs;

  InOut *findInOut(id_type guid);
  static std::string _fname;

public:
  DepManager(JIT &jit);
  void finalizeAndRun();
  ::mlir::OpBuilder &getBuilder() { return _builder; }
  ::mlir::ModuleOp &getmodule() { return _module; }
  ::mlir::Value getDependent(::mlir::OpBuilder &builder,
                             const array_i::future_type &fut);
  ::mlir::Value addDependent(::mlir::OpBuilder &builder, const NDArray *fut);
  void addVal(id_type guid, ::mlir::Value val, SetResFunc cb);
  void addReady(id_type guid, ReadyFunc cb);
  void drop(id_type guid);
  uint64_t handleResult(::mlir::OpBuilder &builder);
  void deliver(std::vector<intptr_t> &, uint64_t);
  std::vector<void *> finalize_inputs();
};

} // namespace jit
} // namespace SHARPY
