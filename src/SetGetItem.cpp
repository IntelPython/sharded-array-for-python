// SPDX-License-Identifier: BSD-3-Clause

/*
  setitem and getitem features.
  Also adds SPMD-like access to data.
*/

#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/CollComm.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Mediator.hpp"
#include "ddptensor/NDSlice.hpp"
#include "ddptensor/Transceiver.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/UtilsAndTypes.hpp"
#include "ddptensor/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/IR/Builders.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace DDPT {

template <typename T> struct mk_array {
  template <typename C> static py::object op(C &&shp, void *&outPtr) {
    auto ary = py::array_t<T>(std::forward<C>(shp));
    outPtr = ary.mutable_data();
    return ary;
  }
};

template <typename T> struct wrap_array {
  template <typename C, typename S>
  static py::object op(C &&shp, S &&str, void *data, const py::handle &handle) {
    return py::array(std::forward<C>(shp), std::forward<S>(str),
                     reinterpret_cast<T *>(data), handle);
  }
};

py::object wrap(DDPTensorImpl::ptr_type tnsr, const py::handle &handle) {
  auto tmp_shp = tnsr->local_shape();
  auto tmp_str = tnsr->local_strides();
  auto nd = tnsr->ndims();
  auto eSz = sizeof_dtype(tnsr->dtype());
  std::vector<ssize_t> strides(nd);
  for (auto i = 0; i < nd; ++i) {
    strides[i] = eSz * tmp_str[i];
  }

  return dispatch<wrap_array>(tnsr->dtype(),
                              std::vector<ssize_t>(tmp_shp, &tmp_shp[nd]),
                              strides, tnsr->data(), handle);
}

// ***************************************************************************

struct DeferredGetLocals
    : public DeferredT<GetItem::py_promise_type, GetItem::py_future_type> {
  id_type _a;
  py::handle _handle;

  DeferredGetLocals() = default;
  DeferredGetLocals(const tensor_i::future_type &a, py::handle &handle)
      : _a(a.guid()), _handle(handle) {}

  void run() override {
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<DDPTensorImpl>(aa);
    assert(a_ptr);
    auto res = wrap(a_ptr, _handle);
    set_value(py::make_tuple(res));
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const { return F_GETLOCALS; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
  }
};

// ***************************************************************************

struct DeferredGather
    : public DeferredT<GetItem::py_promise_type, GetItem::py_future_type> {
  id_type _a;
  rank_type _root;

  DeferredGather() = default;
  DeferredGather(const tensor_i::future_type &a, rank_type root)
      : _a(a.guid()), _root(root) {}

  void run() override {
    // gather
    // We simply create a local buffer, copy our local data to the right place
    // and then call AllGatherV via inplace operation.
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<DDPTensorImpl>(aa);
    assert(a_ptr);
    auto trscvr = a_ptr->transceiver();
    auto myrank = trscvr ? trscvr->rank() : 0;
    bool sendonly = _root != REPLICATED && _root != myrank;

    void *outPtr = nullptr;
    py::object res;
    if (!sendonly || !trscvr) {
      auto tmp = a_ptr->shape();
      std::vector<ssize_t> tmpv(tmp, &tmp[a_ptr->ndims()]);
      // numpy treats 0d arrays as empty arrays, not as a scalar as we do
      if (tmpv.empty()) {
        tmpv.emplace_back(1);
      }
      res = dispatch<mk_array>(a_ptr->dtype(), std::move(tmpv), outPtr);
    }

    gather_tensor(a_ptr, _root, outPtr);

    set_value(res);
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const { return F_GATHER; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
  }
};

// ***************************************************************************

struct DeferredSetItem : public Deferred {
  id_type _a;
  id_type _b;
  NDSlice _slc;

  DeferredSetItem() = default;
  DeferredSetItem(const tensor_i::future_type &a,
                  const tensor_i::future_type &b,
                  const std::vector<py::slice> &v)
      : Deferred(a.guid(), a.dtype(), a.shape(), a.team(), a.balanced()),
        _a(a.guid()), _b(b.guid()), _slc(v) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // get params and extract offsets/sizes/strides
    const auto dtype = this->dtype();
    auto av = dm.getDependent(builder, _a);
    auto bv = dm.getDependent(builder, _b);
    auto &offs = _slc.offsets();
    auto &sizes = _slc.sizes();
    auto &strides = _slc.strides();
    auto nd = offs.size();
    // convert C++ slices into vectors of MLIR Values
    std::vector<::mlir::Value> offsV(nd);
    std::vector<::mlir::Value> sizesV(nd);
    std::vector<::mlir::Value> stridesV(nd);
    for (auto i = 0; i < nd; ++i) {
      offsV[i] = ::imex::createIndex(loc, builder, offs[i]);
      if (sizes[i] == ALL_SIZE) {
        sizesV[i] =
            builder.create<::imex::ptensor::DimOp>(loc, av, i).getResult();
      } else {
        sizesV[i] = ::imex::createIndex(loc, builder, sizes[i]);
      }
      stridesV[i] = ::imex::createIndex(loc, builder, strides[i]);
    }
    // insertsliceop has no return value, so we just create the op...
    (void)builder.create<::imex::ptensor::InsertSliceOp>(loc, av, bv, offsV,
                                                         sizesV, stridesV);
    // ... and use av as to later create the ptensor
    dm.addReady(this->guid(), [this](id_type guid) {
      assert(this->guid() == guid);
      this->set_value(Registry::get(this->_a).get());
    });
    return false;
  }

  FactoryId factory() const { return F_SETITEM; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_b)>(_b);
    ser.template object(_slc);
  }
};

// ***************************************************************************

struct DeferredMap : public Deferred {
  id_type _a;
  py::object _func;

  DeferredMap() = default;
  DeferredMap(const tensor_i::future_type &a, py::object &func)
      : Deferred(a.guid(), a.dtype(), a.shape(), a.team(), a.balanced()),
        _a(a.guid()), _func(func) {}

  void run() override {
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<DDPTensorImpl>(aa);
    assert(a_ptr);
    auto nd = a_ptr->ndims();
    auto lOffs = a_ptr->local_offsets();
    std::vector<int64_t> lIdx(nd);
    std::vector<int64_t> gIdx(nd);

    dispatch(a_ptr->dtype(), a_ptr->data(), [&](auto *ptr) {
      forall(
          0, ptr, a_ptr->local_shape(), a_ptr->local_strides(), nd, lIdx,
          [&](const std::vector<int64_t> &idx, auto *elPtr) {
            for (auto i = 0; i < nd; ++i) {
              gIdx[i] = idx[i] + lOffs[i];
            }
            auto pyIdx = _make_tuple(gIdx);
            *elPtr =
                _func(*pyIdx)
                    .cast<
                        typename std::remove_pointer<decltype(elPtr)>::type>();
          });
    });

    this->set_value(aa);
  };

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const { return F_MAP; }

  template <typename S> void serialize(S &ser) {
    assert(false);
    ser.template value<sizeof(_a)>(_a);
    // nope ser.template value<sizeof(_func)>(_func);
  }
};

// ***************************************************************************

struct DeferredGetItem : public Deferred {
  id_type _a;
  NDSlice _slc;

  static shape_type mkShape(const tensor_i::future_type &a,
                            const NDSlice &slc) {
    shape_type shape;
    for (auto i = 0; i < slc.sizes().size(); ++i) {
      auto s = slc.sizes()[i];
      shape.emplace_back(s == ALL_SIZE ? a.shape()[i] : s);
    }
    return shape;
  }

  DeferredGetItem() = default;
  DeferredGetItem(const tensor_i::future_type &a, NDSlice &&v)
      : Deferred(a.dtype(), std::move(mkShape(a, v)), a.team(), false),
        _a(a.guid()), _slc(std::move(v)) {}

  void run() {
    // const auto a = std::move(Registry::get(_a).get());
    // set_value(std::move(TypeDispatch<x::GetItem>(a, _slc)));
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // get params and extract offsets/sizes/strides
    const auto dtype = this->dtype();
    auto av = dm.getDependent(builder, _a);
    const auto &offs = _slc.offsets();
    const auto &sizes =
        shape(); // we already converted ALL_SIZE as much as posible
    const auto &strides = _slc.strides();
    auto nd = offs.size();
    // convert C++ slices into vectors of MLIR Values
    std::vector<::mlir::OpFoldResult> offsV(nd);
    std::vector<::mlir::OpFoldResult> sizesV(nd);
    std::vector<::mlir::OpFoldResult> stridesV(nd);
    for (auto i = 0; i < nd; ++i) {
      offsV[i] = ::imex::createIndex(loc, builder, offs[i]);
      stridesV[i] = ::imex::createIndex(loc, builder, strides[i]);
      if (sizes[i] < 0) {
        sizesV[i] =
            builder.create<::imex::ptensor::DimOp>(loc, av, i).getResult();
      } else {
        sizesV[i] = builder.getIndexAttr(sizes[i]);
      }
    }

    // auto outnd = nd == 0 || _slc.size() == 1 ? 0 : nd;
    auto outTyp = ::imex::ptensor::PTensorType::get(
        shape(), ::imex::dist::getElementType(av));
    // now we can create the PTensor op using the above Values
    auto res = builder.create<::imex::ptensor::SubviewOp>(
        loc, outTyp, av, offsV, sizesV, stridesV);

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                auto t = mk_tnsr(reinterpret_cast<Transceiver *>(this->team()),
                                 _dtype, this->shape(), l_allocated, l_aligned,
                                 l_offset, l_sizes, l_strides, o_allocated,
                                 o_aligned, o_offset, o_sizes, o_strides,
                                 r_allocated, r_aligned, r_offset, r_sizes,
                                 r_strides, lo_allocated, lo_aligned);
                if (Registry::has(_a)) {
                  t->set_base(Registry::get(_a).get());
                } // else _a is a temporary and was dropped
                this->set_value(std::move(t));
              });
    return false;
  }

  FactoryId factory() const { return F_GETITEM; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template object(_slc);
  }
};

// ***************************************************************************

ddptensor *GetItem::__getitem__(const ddptensor &a,
                                const std::vector<py::slice> &v) {
  NDSlice slc(v);
  return new ddptensor(defer<DeferredGetItem>(a.get(), std::move(slc)));
}

GetItem::py_future_type GetItem::get_locals(const ddptensor &a, py::handle h) {
  return defer<DeferredGetLocals>(a.get(), h);
}

GetItem::py_future_type GetItem::gather(const ddptensor &a, rank_type root) {
  return defer<DeferredGather>(a.get(), root);
}

ddptensor *SetItem::__setitem__(ddptensor &a, const std::vector<py::slice> &v,
                                const py::object &b) {
  auto bb = Creator::mk_future(b, a.get().team(), a.get().dtype());
  a.put(defer<DeferredSetItem>(a.get(), bb.first->get(), v));
  if (bb.second)
    delete bb.first;
  return &a;
}

ddptensor *SetItem::map(ddptensor &a, py::object &b) {
  a.put(defer<DeferredMap>(a.get(), b));
  return &a;
}

py::object GetItem::get_slice(const ddptensor &a,
                              const std::vector<py::slice> &v) {
  const auto aa = std::move(a.get());
  return {}; // FIXME TypeDispatch<x::SPMD>(aa.get(), NDSlice(v), aa.guid());
}

FACTORY_INIT(DeferredGetItem, F_GETITEM);
FACTORY_INIT(DeferredSetItem, F_SETITEM);
FACTORY_INIT(DeferredMap, F_MAP);
FACTORY_INIT(DeferredGather, F_GATHER);
FACTORY_INIT(DeferredGetLocals, F_GETLOCALS);
} // namespace DDPT
