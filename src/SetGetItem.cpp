// SPDX-License-Identifier: BSD-3-Clause

/*
  setitem and getitem features.
  Also adds SPMD-like access to data.
*/

#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/CollComm.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Mediator.hpp"
#include "ddptensor/NDSlice.hpp"
#include "ddptensor/Transceiver.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/UtilsAndTypes.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/IR/Builders.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#if 0
namespace x {

    class GetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(const NDSlice & slice, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            const auto & slc = a_ptr->slice();
            if(slc.ndims() != slice.ndims())
                throw std::runtime_error("Index dimensionality must match array dimensionality");

            return operatorx<T>::mk_tx(*a_ptr.get(), slice.trim(slc.slice()));
        }
    };

    class SetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // copy data from val into (*dest)[slice]
        // this is a non-collective call.
        template<typename T, typename X, typename U>
        static void _set_slice(X && dest, const PVSlice & dest_view, const std::shared_ptr<DPTensorX<U>> & val, const NDSlice & val_slice, id_type val_guid)
        {
            auto nd = dest_view.ndims();
            if(val_slice.size() != dest_view.size())
                throw std::runtime_error("Input and output slices must be of same size");

            // Create a view into val
            PVSlice needed_val_view(val->slice(), val_slice);

            // we can now compute which ranks actually hold which piece of the data from val that we need locally
            for(rank_type i=0; i<getTransceiver()->nranks(); ++i ) {
                // get local view into val
                PVSlice val_local_view(val->slice(), i);
                NDSlice curr_needed_val_slice = needed_val_view.local_slice(i);
                NDSlice curr_local_val_slice = val_local_view.map_slice(curr_needed_val_slice);
                NDSlice curr_needed_norm_slice = needed_val_view.map_slice(curr_needed_val_slice);
                PVSlice my_curr_needed_view = PVSlice(dest_view, curr_needed_norm_slice);
                NDSlice my_curr_local_slice = my_curr_needed_view.tile_slice(getTransceiver()->rank());

                if(curr_needed_norm_slice.size()) {
                    if(i == getTransceiver()->rank()) {
                        // copy locally
                        auto to_v   = xt::strided_view(dest/*.xarray()*/, to_xt(my_curr_local_slice));
                        auto from_v = xt::strided_view(val->xarray(), to_xt(curr_local_val_slice));
                        to_v = from_v;
                    } else {
                        // pull slice directly into new array
                        xt::xarray<U> from_a = xt::empty<U>(curr_local_val_slice.shape());
                        from_a.fill(static_cast<U>(4711));
                        getMediator()->pull(i, val_guid, curr_local_val_slice, from_a.data());
                        auto to_v = xt::strided_view(dest/*.xarray()*/, to_xt(my_curr_local_slice));
                        to_v = from_a;
                    }
                }
            }
        }

        // FIXME We use a generic SPMD/PGAS mechanism to pull elements from remote
        // on all procs simultaneously.  Since __setitem__ is collective we could
        // implement a probaly more efficient mechanism which pushes data and/or using RMA.
        template<typename A, typename B>
        static ptr_type op(const NDSlice & slice, id_type val_guid, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            // Use given slice to create a global view into orig array
            PVSlice g_slc_view(a_ptr->slice(), slice);
            PVSlice my_rel_slice(g_slc_view, getTransceiver()->rank());
            NDSlice my_norm_slice = g_slc_view.map_slice(my_rel_slice.local_slice()); //slice());my_slice);

            if(getTransceiver()->is_spmd()) getTransceiver()->barrier();
            _set_slice<A>(a_ptr->xarray(), my_rel_slice, b_ptr, my_norm_slice, val_guid);
            getTransceiver()->barrier();
            return a_ptr;
        }
    };

    class SPMD
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // get_slice
        template<typename T>
        static py::object op(const NDSlice & slice, id_type val_guid, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto shp = slice.shape();
            auto sz = VPROD(shp);
            auto res = py::array_t<T>(sz);
            auto ax = xt::adapt(res.mutable_data(), sz, xt::no_ownership(), shp);
            PVSlice slc{shp, NOSPLIT};
            SetItem::_set_slice<T>(ax, slc, a_ptr, slice, val_guid);
            return res;
        }

        // get_local
        template<typename T>
        static py::object op(py::handle & handle, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto slc = a_ptr->slice().tile_slice();
            auto tshp = a_ptr->slice().tile_shape();
            auto nd = slc.ndims();
             // buffer protocol accepts strides in number of bytes not elements!
            std::vector<uint64_t> strides(nd, sizeof(T));
            uint64_t off = slc.dim(nd-1)._start * sizeof(T); // start offset
            for(int i=nd-2; i>=0; --i) {
                auto slci = slc.dim(i);
                auto tmp = strides[i+1] * tshp[i+1];
                strides[i] = slci._step * tmp;
                off += slci._start * tmp;
            }
            off /= sizeof(T); // we need the offset in number of elements
            strides.back() = slc.dim(nd-1)._step * sizeof(T);
            T * data = a_ptr->xarray().data();
            return py::array(std::move(slc.shape()), std::move(strides), data + off, handle);
        }
    };

} // namespace x
#endif // if 0

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

// ***************************************************************************

struct DeferredGetLocal
    : public DeferredT<GetItem::py_promise_type, GetItem::py_future_type> {
  id_type _a;
  py::handle _handle;

  DeferredGetLocal() = default;
  DeferredGetLocal(const tensor_i::future_type &a, py::handle &handle)
      : _a(a.id()), _handle(handle) {}

  void run() override {
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<DDPTensorImpl>(aa);
    assert(a_ptr);
    auto tmp_shp = a_ptr->local_shape();
    auto tmp_str = a_ptr->local_strides();
    auto nd = a_ptr->ndims();
    auto eSz = sizeof_dtype(a_ptr->dtype());
    std::vector<ssize_t> strides(nd);
    for (auto i = 0; i < nd; ++i) {
      strides[i] = eSz * tmp_str[i];
    }
    auto res = dispatch<wrap_array>(a_ptr->dtype(),
                                    std::vector<ssize_t>(tmp_shp, &tmp_shp[nd]),
                                    strides, a_ptr->data(), _handle);
    set_value(res);
  }

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const { return F_GETLOCAL; }

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
      : _a(a.id()), _root(root) {}

  void run() override {
    // gather
    // We simply create a local buffer, copy our local data to the right place
    // and then call AllGatherV via inplace operation.
    auto trscvr = getTransceiver();
    auto myrank = trscvr->rank();
    auto aa = std::move(Registry::get(_a).get());
    auto a_ptr = std::dynamic_pointer_cast<DDPTensorImpl>(aa);
    assert(a_ptr);
    bool sendonly = _root != REPLICATED && _root != myrank;

    void *outPtr = nullptr;
    py::object res;
    if (!sendonly) {
      auto tmp = a_ptr->shape();
      // std::vector<ssize_t> shp(tmp, &tmp[a_ptr->ndims()]);
      res = dispatch<mk_array>(a_ptr->dtype(),
                               std::vector<ssize_t>(tmp, &tmp[a_ptr->ndims()]),
                               outPtr);
      // (void*)nullptr, [&shp, &res, &outPtr](auto * ptr) {
      //     auto ary = py::array_t<double>({4,4});
      //     res = ary;
      //     outPtr = ary.mutable_data();
      // });
    }

    gather_tensor(a_ptr, _root, outPtr);

    set_value(res);
  }

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
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
      : Deferred(a.id(), a.dtype(), a.rank(), a.balanced()), _a(a.id()),
        _b(b.id()), _slc(v) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
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
      sizesV[i] = ::imex::createIndex(loc, builder, sizes[i]);
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

struct DeferredGetItem : public Deferred {
  id_type _a;
  NDSlice _slc;

  DeferredGetItem() = default;
  DeferredGetItem(const tensor_i::future_type &a,
                  const std::vector<py::slice> &v)
      : Deferred(a.dtype(), a.rank(), false), _a(a.id()), _slc(v) {}

  void run() {
    // const auto a = std::move(Registry::get(_a).get());
    // set_value(std::move(TypeDispatch<x::GetItem>(a, _slc)));
  }

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // get params and extract offsets/sizes/strides
    const auto dtype = this->dtype();
    auto av = dm.getDependent(builder, _a);
    auto &offs = _slc.offsets();
    auto &sizes = _slc.sizes();
    auto &strides = _slc.strides();
    auto nd = offs.size();
    // convert C++ slices into vectors of MLIR Values
    std::vector<::mlir::OpFoldResult> offsV(nd);
    std::vector<::mlir::OpFoldResult> sizesV(nd);
    std::vector<::mlir::OpFoldResult> stridesV(nd);
    ::mlir::SmallVector<int64_t> shape(nd, ::mlir::ShapedType::kDynamic);
    for (auto i = 0; i < nd; ++i) {
      offsV[i] = ::imex::createIndex(loc, builder, offs[i]);
      stridesV[i] = ::imex::createIndex(loc, builder, strides[i]);
      if (sizes[i] == 1) {
        sizesV[i] = builder.getIndexAttr(sizes[i]);
        shape[i] = sizes[i];
      } else {
        sizesV[i] = ::imex::createIndex(loc, builder, sizes[i]);
      }
    }

    auto oTyp = ::imex::dist::getPTensorType(av);
    // auto outnd = nd == 0 || _slc.size() == 1 ? 0 : nd;
    auto outTyp =
        ::imex::ptensor::PTensorType::get(shape, oTyp.getElementType());
    // if(auto dtyp = av.getType().dyn_cast<::imex::dist::DistTensorType>()) {
    //   av = builder.create<::mlir::UnrealizedConversionCastOp>(loc,
    //   dtyp.getPTensorType(), av).getResult(0);
    // }
    // now we can create the PTensor op using the above Values
    auto res = builder.create<::imex::ptensor::SubviewOp>(
        loc, outTyp, av, offsV, sizesV, stridesV);
    dm.addVal(this->guid(), res,
              [this, dtype](Transceiver *transceiver, uint64_t rank,
                            void *allocated, void *aligned, intptr_t offset,
                            const intptr_t *sizes, const intptr_t *strides,
                            uint64_t *gs_allocated, uint64_t *gs_aligned,
                            uint64_t *lo_allocated, uint64_t *lo_aligned,
                            uint64_t balanced) {
                this->set_value(std::move(
                    mk_tnsr(transceiver, dtype, rank, allocated, aligned,
                            offset, sizes, strides, gs_allocated, gs_aligned,
                            lo_allocated, lo_aligned, balanced)));
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
  return new ddptensor(defer<DeferredGetItem>(a.get(), v));
}

GetItem::py_future_type GetItem::get_local(const ddptensor &a, py::handle h) {
  return defer<DeferredGetLocal>(a.get(), h);
}

GetItem::py_future_type GetItem::gather(const ddptensor &a, rank_type root) {
  return defer<DeferredGather>(a.get(), root);
}

ddptensor *SetItem::__setitem__(ddptensor &a, const std::vector<py::slice> &v,
                                const py::object &b) {

  auto bb = Creator::mk_future(b);
  a.put(defer<DeferredSetItem>(a.get(), bb.first->get(), v));
  if (bb.second)
    delete bb.first;
  return &a;
}

py::object GetItem::get_slice(const ddptensor &a,
                              const std::vector<py::slice> &v) {
  const auto aa = std::move(a.get());
  return {}; // FIXME TypeDispatch<x::SPMD>(aa.get(), NDSlice(v), aa.id());
}

FACTORY_INIT(DeferredGetItem, F_GETITEM);
FACTORY_INIT(DeferredSetItem, F_SETITEM);
FACTORY_INIT(DeferredGather, F_GATHER);
FACTORY_INIT(DeferredGather, F_GETLOCAL);
