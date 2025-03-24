// SPDX-License-Identifier: BSD-3-Clause

#include "sharpy/jit/DepManager.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/UtilsAndTypes.hpp"
#include "sharpy/itac.hpp"
#include <cstdlib>
#include <iostream>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <string>

namespace SHARPY {
namespace jit {

static bool FORCE_DIST = get_bool_env("SHARPY_FORCE_DIST");
std::string DepManager::_fname = "sharpy_jit";

static void
mkDLTIAttr(mlir::ImplicitLocOpBuilder b, mlir::Operation *op,
           mlir::ArrayRef<std::pair<std::string, mlir::Attribute>> vals) {
  mlir::SmallVector<mlir::DataLayoutEntryInterface> entries;
  for (auto v : vals) {
    entries.emplace_back(
        mlir::DataLayoutEntryAttr::get(b.getStringAttr(v.first), v.second));
  }
  op->setAttr(mlir::DLTIDialect::kMapAttrName,
              mlir::MapAttr::get(b.getContext(), entries));
}

DepManager::InOut::InOut(DepManager *dm, id_type guid,
                         const ::mlir::Value &value,
                         const SetResFunc &setResFunc, Deferred *deferred,
                         id_type aliasOf)
    : _guid(guid), _aliasOf(aliasOf), _value(value), _setResFunc(setResFunc),
      _deferred(deferred) {
  // find the root base, assign it base and increase its alias count
  if (aliasOf != NOGUID) {
    InOut *base = this;
    while (base->_aliasOf != NOGUID) {
      base = dm->findInOut(base->_aliasOf);
      assert(base);
    }
    ++base->_numAliases;
    this->_aliasOf = base->_guid;
  }
}

DepManager::DepManager(jit::JIT &jit)
    : _jit(jit),
      _builder(mlir::UnknownLoc::get(&jit.context()), &jit.context()) {
  auto loc = _builder.getLoc();
  _module = _builder.create<::mlir::ModuleOp>(loc);
  mkDLTIAttr(_builder, _module,
             {{"MPI:Implementation", _builder.getStringAttr("mpich")},
              {"MPI:comm_world_rank",
               _builder.getI32IntegerAttr(getTransceiver()->rank())}});
  auto dummyFuncType = _builder.getFunctionType({}, {});
  _func = _builder.create<::mlir::func::FuncOp>(_fname, dummyFuncType);
  auto &entryBlock = *_func.addEntryBlock();
  _builder.setInsertionPointToStart(&entryBlock);
}

void DepManager::finalizeAndRun() {
  auto input = finalize_inputs();
  uint64_t osz = handleResult(_builder);
  _func->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                 _builder.getUnitAttr());
  if (_jit.verbose())
    _func.getFunctionType().dump();
  if (getTransceiver()) {
    int64_t nRanks = (int64_t)getTransceiver()->nranks();
    auto mesh = getTransceiver()->mesh();
    mlir::ImplicitLocOpBuilder::InsertionGuard g(_builder);
    _builder.setInsertionPointToStart(&_module.getRegion().front());
    auto meshOp = _builder.create<::mlir::mesh::MeshOp>(
        _builder.getUnknownLoc(), mesh, mlir::ArrayRef<int64_t>{nRanks});
    meshOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  }
  _module.push_back(_func);
  if (osz > 0 || !input.empty()) {
    auto output = _jit.run(_module, _fname, input, osz);
    for (auto p : input) {
      delete[] reinterpret_cast<intptr_t *>(p);
    }
    if (output.size() != osz)
      throw std::runtime_error("failed running jit");
    deliver(output, osz);
  } else {
    if (_jit.verbose())
      std::cerr << "\tskipping\n";
    deliver({}, 0);
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

::mlir::Value DepManager::getDependent(mlir::ImplicitLocOpBuilder &builder,
                                       const array_i::future_type &fut) {
  id_type guid = fut.guid();
  if (auto d = findInOut(guid)) {
    return d->_value;
  } else {
    auto impl = std::dynamic_pointer_cast<NDArray>(fut.get());
    return addDependent(builder, impl.get(), guid);
  }
};

static ::mlir::RankedTensorType getTensorType(const shape_type &shape,
                                              ::mlir::Type elType) {
  return mlir::RankedTensorType::get(shape, elType);
}

::mlir::Value DepManager::addDependent(mlir::ImplicitLocOpBuilder &builder,
                                       const NDArray *impl, id_type guid) {
  if (findInOut(guid)) {
    throw std::runtime_error("Internal error: array already added");
  }
  auto idx = _lastIn;
  size_t ndims = impl->ndims();
  ::mlir::SmallVector<int64_t> zeros(ndims, 0);
  auto elType(getMLIRType(builder, impl->dtype()));

  auto storeMR = [ndims](const DynMemRef &mr) -> intptr_t * {
    intptr_t *buff = new intptr_t[memref_sz(ndims)];
    buff[0] = reinterpret_cast<intptr_t>(mr._allocated);
    buff[1] = reinterpret_cast<intptr_t>(mr._aligned);
    buff[2] = static_cast<intptr_t>(mr._offset);
    memcpy(buff + 3, mr._sizes, ndims * sizeof(intptr_t));
    memcpy(buff + 3 + ndims, mr._strides, ndims * sizeof(intptr_t));
    return buff;
  };

  auto loc = builder.getUnknownLoc();
  mlir::ImplicitLocOpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&_func.front());

  auto typ = getTensorType(impl->shape(), elType);
  _func.insertArgument(idx, typ, {}, loc);
  _inputs.push_back(storeMR(impl->owned_data()));
  auto arg = shardNow(builder, _func.getArgument(idx), impl->team(),
                      impl->split_axes(), impl->halo_sizes(),
                      impl->sharded_dims_offsets());
  _inOut.emplace_back(InOut(guid, arg));
  _lastIn += 1;

  return arg;
}

std::vector<void *> DepManager::finalize_inputs() {
  _lastIn = 0;
  return std::move(_inputs);
}

void DepManager::addVal(Deferred *deferred, ::mlir::Value val, SetResFunc cb,
                        id_type aliasOf) {
  auto guid = deferred->guid();
  if (findInOut(guid)) {
    throw std::runtime_error("Internal error: array already added");
  }
  auto tmp = _inOut.emplace_back(InOut(this, guid, val, cb, deferred, aliasOf));
}

void DepManager::addReady(id_type guid, ReadyFunc cb) {
  auto x = findInOut(guid);
  if (!x) {
    x = &_inOut.emplace_back(InOut{guid});
  }
  x->_readyFuncs.emplace_back(cb);
}

void DepManager::drop(id_type guid) {
  if (auto x = findInOut(guid)) {
    if (!x->_isAlive) {
      return;
    }
    x->_isAlive = false;
    if (x->_aliasOf != NOGUID) {
      auto b = findInOut(x->_aliasOf);
      assert(b);
      --b->_numAliases;
    }
  }
}

uint64_t DepManager::handleResult(mlir::ImplicitLocOpBuilder &builder) {
  mlir::ImplicitLocOpBuilder::InsertionGuard guard(builder);
  std::vector<::mlir::Value> ret_values;
  uint64_t sz = 0;
  unsigned idx = 0;

  for (auto &x : _inOut) {
    if (x.isResult()) {
      auto rank =
          mlir::cast<::mlir::RankedTensorType>(x._value.getType()).getRank();
      bool isDist = FORCE_DIST || (rank > 0 && getTransceiver()->nranks() > 1);
      ret_values.emplace_back(x._value);
      _func.insertResult(idx++, x._value.getType(), {});
      if (isDist) {
        builder.setInsertionPointAfterValue(x._value);
        auto sharding = builder.create<mlir::mesh::GetShardingOp>(x._value);
        ret_values.emplace_back(sharding);
        _func.insertResult(idx++, sharding.getType(), {});
      }
      x._rank = rank;
      x._isDist = isDist;
      sz += ndarray_sz(rank, isDist);
    }
  }
  if (HAS_ITAC()) {
    int vtExeSym = 0, vtSHARPYClass = 0;
    VT(VT_classdef, "sharpy", &vtSHARPYClass);
    VT(VT_funcdef, "execute", vtSHARPYClass, &vtExeSym);
    ::mlir::Value s = builder.create<::mlir::arith::ConstantOp>(
        builder.getI32IntegerAttr(vtExeSym));
    auto end = builder.create<::mlir::func::CallOp>(
        builder.getUnknownLoc(), "VT_end",
        ::mlir::TypeRange(builder.getIntegerType(32)), ::mlir::ValueRange(s));
    mlir::ImplicitLocOpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(end->getBlock());
    (void)builder.create<::mlir::func::CallOp>(
        builder.getUnknownLoc(), "VT_begin",
        ::mlir::TypeRange(builder.getIntegerType(32)), ::mlir::ValueRange(s));
  }
  (void)builder.create<::mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                               ret_values);
  return 2 * sz;
}

void DepManager::deliver(const std::vector<intptr_t> &output, uint64_t sz) {
  size_t pos = 0;
  for (auto &x : _inOut) {
    if (x.isResult()) {
      assert(output.size() && sz);
      auto get_dynmemref = [](const intptr_t *buff, size_t &pos, int rank,
                              bool enabled) {
        if (!enabled) {
          return DynMemRef();
        }
        auto t_allocated = reinterpret_cast<intptr_t *>(buff[pos]);
        auto t_aligned = reinterpret_cast<intptr_t *>(buff[pos + 1]);
        intptr_t t_offset = buff[pos + 2];
        auto t_sizes = &buff[pos + 3];
        auto t_strides = &buff[pos + 3 + rank];
        pos += memref_sz(rank);
        return DynMemRef(rank, t_allocated, t_aligned, t_offset, t_sizes,
                         t_strides);
      };
      auto data = get_dynmemref(output.data(), pos, x._rank, true);
      auto splits = get_dynmemref(output.data(), pos, 2, x._isDist);
      auto halos = get_dynmemref(output.data(), pos, 2, x._isDist);
      auto offs = get_dynmemref(output.data(), pos, 2, x._isDist);

      auto base = NOGUID;
      if (x._aliasOf != NOGUID) {
        auto b = findInOut(x._aliasOf);
        assert(b);
        if (b->_isAlive || b->_numAliases) {
          base = x._aliasOf;
        }
      }
      x._setResFunc(x._deferred, x._rank, std::move(data), std::move(splits),
                    std::move(halos), std::move(offs), base);
      if (x._rank > 0 && x._isDist) {
      }
    }
  }
  for (auto &x : _inOut) {
    for (auto cb : x._readyFuncs) {
      cb(x._guid);
    }
  }
  _inOut.clear();
}

} // namespace jit
} // namespace SHARPY
