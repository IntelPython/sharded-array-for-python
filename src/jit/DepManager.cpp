// SPDX-License-Identifier: BSD-3-Clause

#include "sharpy/jit/DepManager.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/UtilsAndTypes.hpp"
#include "sharpy/itac.hpp"
#include <cstdlib>
#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <string>

namespace SHARPY {
namespace jit {

std::string DepManager::_fname = "sharpy_jit";

DepManager::DepManager(jit::JIT &jit) : _jit(jit), _builder(&jit.context()) {
  auto loc = _builder.getUnknownLoc();
  _module = _builder.create<::mlir::ModuleOp>(loc);
  auto dummyFuncType = _builder.getFunctionType({}, {});
  _func = _builder.create<::mlir::func::FuncOp>(loc, _fname, dummyFuncType);
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
    ::mlir::OpBuilder::InsertionGuard g(_builder);
    _builder.setInsertionPointToStart(&_module.getRegion().front());
    auto meshOp = _builder.create<::mlir::mesh::MeshOp>(
        _builder.getUnknownLoc(), mesh, mlir::ArrayRef<int64_t>{nRanks});
    meshOp.setVisibility(mlir::SymbolTable::Visibility::Private);
    (void)_builder.create<::mlir::memref::GlobalOp>(
        _builder.getUnknownLoc(),
        /*sym_name=*/"static_mpi_rank",
        /*sym_visibility=*/_builder.getStringAttr("public"),
        /*type=*/mlir::MemRefType::get({}, _builder.getIndexType()),
        /*initial_value=*/
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get({}, _builder.getIndexType()),
            {getTransceiver()->rank()}),
        /*constant=*/true,
        /*alignment=*/mlir::IntegerAttr());
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

::mlir::Value DepManager::addDependent(::mlir::OpBuilder &builder,
                                       const NDArray *impl) {
  id_type guid = impl->guid();
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
  };

  auto typ = getTensorType(ndims, impl->owned_data(), elType);
  _func.insertArgument(idx, typ, {}, loc);
  _inputs.push_back(storeMR(impl->owned_data()));
  auto arg = shardNow(builder, loc, _func.getArgument(idx), impl->team());
  _inOut.emplace_back(InOut(guid, arg));
  _lastIn += 1;

  return arg;
}

std::vector<void *> DepManager::finalize_inputs() {
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
  if (x) {
    x->_value = nullptr;
    x->_setResFunc = nullptr;
  }
}

uint64_t DepManager::handleResult(::mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  std::vector<::mlir::Value> ret_values;
  auto loc = builder.getUnknownLoc();
  uint64_t sz = 0;
  unsigned idx = 0;
  for (auto &x : _inOut) {
    ::mlir::Value value = x._value;
    if (value) {
      bool isDist = false;
      auto rank =
          mlir::cast<::mlir::RankedTensorType>(value.getType()).getRank();
      ret_values.emplace_back(value);
      _func.insertResult(idx, value.getType(), {});
      x._rank = rank;
      x._isDist = isDist;
      sz += ndarray_sz(rank, isDist);
      ++idx;
      if (isDist && rank) {
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
                      t_strides, {});
      }
      if (x._rank > 0 && x._isDist) {
      }
    }
    if (!x._readyFuncs.empty()) {
      for (auto cb : x._readyFuncs) {
        cb(x._guid);
      }
    }
  }
  _inOut.clear();
}

} // namespace jit
} // namespace SHARPY
