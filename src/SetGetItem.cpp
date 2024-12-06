// SPDX-License-Identifier: BSD-3-Clause

/*
  setitem and getitem features.
  Also adds SPMD-like access to data.
*/

#include "sharpy/SetGetItem.hpp"
#include "sharpy/CollComm.hpp"
#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/Mediator.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/NDSlice.hpp"
#include "sharpy/Transceiver.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/UtilsAndTypes.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/IR/Builders.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace SHARPY {

template <typename T> struct mk_array {
  template <typename C> static py::handle op(C &&shp, void *&outPtr) {
    auto ary = py::array_t<T>(std::forward<C>(shp));
    outPtr = ary.mutable_data();
    return ary.release();
  }
};

template <typename T> struct wrap_array {
  template <typename C, typename S>
  static py::handle op(C &&shp, S &&str, void *data, const py::handle &handle) {
    return py::array(std::forward<C>(shp), std::forward<S>(str),
                     reinterpret_cast<T *>(data), handle)
        .release();
  }
};

py::handle wrap(NDArray::ptr_type tnsr, const py::handle &handle) {
  auto tmp_shp = tnsr->local_shape();
  auto tmp_str = tnsr->local_strides();
  auto nd = tnsr->ndims();
  int64_t eSz = sizeof_dtype(tnsr->dtype());
  std::vector<ssize_t> strides(nd);
  for (auto i = 0; i < nd; ++i) {
    strides[i] = eSz * tmp_str[i];
    if (strides[i] / tmp_str[i] != eSz) {
      throw std::overflow_error("Fatal: Integer overflow.");
    }
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
  DeferredGetLocals(const array_i::future_type &a, py::handle &handle)
      : _a(a.guid()), _handle(handle) {
    py::gil_scoped_acquire acquire;
    _handle.inc_ref();
  }
  ~DeferredGetLocals() {
    py::gil_scoped_acquire acquire;
    _handle.dec_ref();
  }

  void run() override {
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<NDArray>(aa);
    if (!a_ptr) {
      throw std::invalid_argument("Expected NDArray in getlocals.");
    }
    auto res = wrap(a_ptr, _handle);
    auto tpl = py::make_tuple(py::reinterpret_steal<py::object>(res));
    set_value(tpl.release());
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const override { return F_GETLOCALS; }

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
  DeferredGather(const array_i::future_type &a, rank_type root)
      : _a(a.guid()), _root(root) {}

  void run() override {
    // gather
    // We simply create a local buffer, copy our local data to the right place
    // and then call AllGatherV via inplace operation.
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<NDArray>(aa);
    if (!a_ptr) {
      throw std::invalid_argument("Expected NDArray in gather.");
    }
    auto trscvr = a_ptr->transceiver();
    auto myrank = trscvr ? trscvr->rank() : 0;
    bool sendonly = _root != REPLICATED && _root != myrank;

    void *outPtr = nullptr;
    py::handle res;
    if (!sendonly || !trscvr) {
      std::vector<ssize_t> shp(a_ptr->shape());
      res = dispatch<mk_array>(a_ptr->dtype(), std::move(shp), outPtr);
    }

    gather_array(a_ptr, _root, outPtr);

    set_value(res);
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const override { return F_GATHER; }

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
  DeferredSetItem(const array_i::future_type &a, const array_i::future_type &b,
                  const std::vector<py::slice> &v)
      : Deferred(a.dtype(), a.shape(), a.device(), a.team(), a.guid()),
        _a(a.guid()), _b(b.guid()), _slc(v, a.shape()) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // get params and extract offsets/sizes/strides
    auto av = dm.getDependent(builder, Registry::get(_a));
    auto bv = dm.getDependent(builder, Registry::get(_b));
    auto &offs = _slc.offsets();
    auto &sizes = _slc.sizes();
    auto &strides = _slc.strides();

    // insertsliceop has no return value, so we just create the op...
    (void)builder.create<::imex::ndarray::InsertSliceOp>(loc, av, bv, offs,
                                                         sizes, strides);
    // ... and use av as to later create the ndarray
    dm.addReady(this->guid(), [this](id_type guid) {
      assert(this->guid() == guid);
      this->set_value(Registry::get(this->_a).get());
    });
    return false;
  }

  FactoryId factory() const override { return F_SETITEM; }

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
  DeferredMap(const array_i::future_type &a, py::object &func)
      : Deferred(a.dtype(), a.shape(), a.device(), a.team(), a.guid()),
        _a(a.guid()), _func(func) {}

  void run() override {
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<NDArray>(aa);
    if (!a_ptr) {
      throw std::invalid_argument("Expected NDArray in map.");
    }
    auto nd = a_ptr->ndims();
    auto lOffs = a_ptr->local_offsets();
    std::vector<int64_t> lIdx(nd);
    std::vector<int64_t> gIdx(nd);

    dispatch(a_ptr->dtype(), a_ptr->data(), [&](auto *ptr) {
      forall(
          0, ptr, a_ptr->local_shape(), a_ptr->local_strides(), nd, lIdx,
          [&](const std::vector<int64_t> &idx, auto *elPtr) {
            for (auto i = 0; i < nd; ++i) {
              gIdx[i] = lOffs.empty() ? idx[i] : idx[i] + lOffs[i];
              if (gIdx[i] < idx[i]) {
                throw std::overflow_error("Fatal: Integer overflow in map.");
              }
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

  FactoryId factory() const override { return F_MAP; }

  template <typename S> void serialize(S &ser) {
    throw std::runtime_error("Not implemented");
    ser.template value<sizeof(_a)>(_a);
    // nope ser.template value<sizeof(_func)>(_func);
  }
};

// ***************************************************************************

struct DeferredGetItem : public Deferred {
  id_type _a;
  NDSlice _slc;

  DeferredGetItem() = default;
  DeferredGetItem(const array_i::future_type &a, NDSlice &&v)
      : Deferred(a.dtype(), shape_type(v.sizes()), a.device(), a.team()),
        _a(a.guid()), _slc(std::move(v)) {}

  void run() override {
    // const auto a = std::move(Registry::get(_a).get());
    // set_value(std::move(TypeDispatch<x::GetItem>(a, _slc)));
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // get params and extract offsets/sizes/strides
    auto av = dm.getDependent(builder, Registry::get(_a));
    const auto &offs = _slc.offsets();
    const auto &sizes = shape();
    const auto &strides = _slc.strides();
    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(av.getType());
    auto outTyp = ::mlir::cast<::mlir::RankedTensorType>(
        aTyp.cloneWith(shape(), aTyp.getElementType()));

    // now we can create the NDArray op using the above Values
    auto res = builder.create<::imex::ndarray::SubviewOp>(loc, outTyp, av, offs,
                                                          sizes, strides);

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                auto t =
                    mk_tnsr(this->guid(), _dtype, this->shape(), this->device(),
                            this->team(), allocated, aligned, offset, sizes,
                            strides, std::move(loffs));
                if (Registry::has(_a)) {
                  t->set_base(Registry::get(_a).get());
                } // else _a is a temporary and was dropped
                this->set_value(std::move(t));
              });
    return false;
  }

  FactoryId factory() const override { return F_GETITEM; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template object(_slc);
  }
};

// ***************************************************************************

// extract "start", "stop", "step" int attrs from py::slice
std::optional<int> getSliceAttr(const py::slice &slice,
                                const std::string &name) {
  auto obj = getattr(slice, name.c_str());
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  } else if (py::isinstance<py::int_>(obj)) {
    return std::optional<int>{obj.cast<int>()};
  } else {
    throw std::invalid_argument("Invalid indices");
  }
};

// check that multi-dimensional slice start does not exceed given shape
void validateSlice(const shape_type &shape,
                   const std::vector<py::slice> &slices) {
  auto dim = shape.size();
  for (std::size_t i = 0; i < dim; i++) {
    auto start = getSliceAttr(slices[i], "start");
    if (start && start.value() >= shape[i]) {
      std::stringstream msg;
      msg << "index " << start.value() << " is out of bounds for axis " << i
          << " with size " << shape[i] << "\n";
      throw std::out_of_range(msg.str());
    }
  }
}

FutureArray *GetItem::__getitem__(const FutureArray &a,
                                  const std::vector<py::slice> &v) {
  auto afut = a.get();
  validateSlice(afut.shape(), v);
  NDSlice slc(v, afut.shape());
  return new FutureArray(defer<DeferredGetItem>(afut, std::move(slc)));
}

GetItem::py_future_type GetItem::get_locals(const FutureArray &a,
                                            py::handle h) {
  return defer<DeferredGetLocals>(a.get(), h);
}

GetItem::py_future_type GetItem::gather(const FutureArray &a, rank_type root) {
  return defer<DeferredGather>(a.get(), root);
}

FutureArray *SetItem::__setitem__(FutureArray &a,
                                  const std::vector<py::slice> &v,
                                  const py::object &b) {
  auto afut = a.get();
  validateSlice(afut.shape(), v);
  auto bb = Creator::mk_future(b, afut.device(), afut.team(), afut.dtype());
  a.put(defer<DeferredSetItem>(afut, bb.first->get(), v));
  if (bb.second)
    delete bb.first;
  return &a;
}

FutureArray *SetItem::map(FutureArray &a, py::object &b) {
  a.put(defer<DeferredMap>(a.get(), b));
  return &a;
}

py::object GetItem::get_slice(const FutureArray &a,
                              const std::vector<py::slice> &v) {
  const auto aa = std::move(a.get());
  return {}; // FIXME TypeDispatch<x::SPMD>(aa.get(), NDSlice(v), aa.guid());
}

FACTORY_INIT(DeferredGetItem, F_GETITEM);
FACTORY_INIT(DeferredSetItem, F_SETITEM);
FACTORY_INIT(DeferredMap, F_MAP);
FACTORY_INIT(DeferredGather, F_GATHER);
FACTORY_INIT(DeferredGetLocals, F_GETLOCALS);
} // namespace SHARPY
