// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <sharpy/CppTypes.hpp>
#include <sharpy/MemRefType.hpp>
#include <sharpy/array_i.hpp>
#include <sharpy/jit/mlir.hpp>
#include <vector>

namespace SHARPY {

class Deferred;

namespace jit {

// function type used for reporting back array results generated
// by Deferred::generate_mlir
using SetResFunc = std::function<void(
    Deferred *deferred, uint64_t rank, DynMemRef &&data, DynMemRef &&splits,
    DynMemRef &&halos, DynMemRef &&offs, id_type base)>;
using ReadyFunc = std::function<void(id_type guid)>;

class DepManager {
private:
  struct InOut {
    id_type _guid = 0;
    id_type _aliasOf = NOGUID;
    ::mlir::Value _value = nullptr;
    SetResFunc _setResFunc;
    Deferred *_deferred = nullptr;
    std::vector<ReadyFunc> _readyFuncs;
    int _rank = 0;
    int _numAliases = 0;
    bool _isDist = false;
    bool _isAlive = true;
    InOut(id_type guid = 0, const ::mlir::Value &value = nullptr)
        : _guid(guid), _value(value) {}
    InOut(DepManager *dm, id_type guid = 0,
          const ::mlir::Value &value = nullptr,
          const SetResFunc &setResFunc = nullptr, Deferred *deferred = nullptr,
          id_type aliasOf = NOGUID);
    bool isResult() const { return _setResFunc && (_isAlive || _numAliases); }
  };
  using InOutList = std::vector<InOut>;

  InOutList _inOut;
  int _lastIn = 0;
  JIT &_jit;
  mlir::ImplicitLocOpBuilder _builder;
  ::mlir::ModuleOp _module;
  ::mlir::func::FuncOp _func;
  std::vector<void *> _inputs;

  InOut *findInOut(id_type guid);
  static std::string _fname;

public:
  DepManager(JIT &jit);
  void finalizeAndRun();
  mlir::ImplicitLocOpBuilder &getBuilder() { return _builder; }
  ::mlir::ModuleOp &getmodule() { return _module; }
  ::mlir::Value getDependent(mlir::ImplicitLocOpBuilder &builder,
                             const array_i::future_type &fut);
  ::mlir::Value addDependent(mlir::ImplicitLocOpBuilder &builder,
                             const NDArray *fut, id_type guid);
  void addVal(Deferred *deferred, ::mlir::Value val, SetResFunc cb,
              id_type aliasOf = NOGUID);
  void addReady(id_type guid, ReadyFunc cb);
  // return true if guid was found and dropped
  void drop(id_type guid);
  uint64_t handleResult(mlir::ImplicitLocOpBuilder &builder);
  void deliver(const std::vector<intptr_t> &, uint64_t);
  std::vector<void *> finalize_inputs();
};

} // namespace jit
} // namespace SHARPY
