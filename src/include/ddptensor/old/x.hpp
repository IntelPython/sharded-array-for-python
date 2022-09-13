// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i plus compute kernels using XTensor.
// Interfaces are based on shared_ptr<tensor_i>.
// To make sure we use the correctly typed implementations we dynamic_cast
// input tensors.
// Many kernels life in a single function. This function then accepts an
// operation identifiyer (enum value) and dispatches accordingly.

#pragma once

#include <type_traits>
#include <sstream>
#include <memory>
#include <algorithm>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xio.hpp>
#include <pybind11/numpy.h>
#include "PVSlice.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"
#include "Deferred.hpp"

namespace x
{
    template<typename T>
    T to_native(const py::object & o)
    {
        return o.cast<T>();
    }

    inline xt::xstrided_slice_vector to_xt(const NDSlice & slice)
    {
        xt::xstrided_slice_vector sv;
        for(auto s : slice.slices()) sv.push_back(xt::range(s._start, s._end, s._step));
        return sv;
    }

    using DPTensorBaseX = tensor_i;

    template<typename T>
    class DPTensorX : public DPTensorBaseX
    {
        mutable rank_type _owner;
        PVSlice _slice;
        xt::xstrided_slice_vector _lslice;
        std::shared_ptr<xt::xarray<T>> _xarray;
        mutable T _replica = 0;

    public:
        using typed_ptr_type = std::shared_ptr<DPTensorX<T>>;

        DPTensorX(const DPTensorX<T> &) = delete;
        DPTensorX(DPTensorX<T> &&) = default;

        template<typename I>
        DPTensorX(PVSlice && slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(std::move(slc)), // static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              _lslice(to_xt(_slice.tile_slice())),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
            if(owner != NOOWNER)
                throw(std::runtime_error("Creating from PVSlice must be NOOWNER"));
        }

        template<typename I>
        DPTensorX(const shape_type & slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(slc, static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
        }

        template<typename I>
        DPTensorX(shape_type && shp, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(std::move(shp), static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
        }

        DPTensorX(const shape_type & shp, T * ptr, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(shp, static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              _xarray(std::make_shared<xt::xarray<T>>(xt::adapt(ptr, VPROD(shp), xt::no_ownership(), shp)))
        {
        }

        DPTensorX(const shape_type & shp, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(shp, static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              _xarray(std::make_shared<xt::xarray<T>>(xt::empty<T>(_slice.tile_shape())))
        {
        }

        DPTensorX(const T & v, rank_type owner=getTransceiver()->rank())
            : _owner(owner),
              _slice(shape_type{1}, static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
              // _lslice({xt::newaxis()}), //to_xt(_slice.slice())),
              _xarray(std::make_shared<xt::xarray<T>>(1)),
              _replica(v)
        {
            *_xarray = v;
        }

        template<typename O>
        DPTensorX(const DPTensorX<O> & org, const NDSlice & slc, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(org._slice, slc),
              _lslice(to_xt(_slice.tile_slice())),
              _xarray(org._xarray)
        {
            if(owner == NOOWNER && slice().size() <= 1) {
                set_owner(org.slice().owner(slc));
            } else if(owner == REPLICATED) {
                _replica = *(xt::strided_view(xarray(), to_xt(slice().slice())).begin());
            }
        }

        template<typename O>
        DPTensorX(O && org, PVSlice && slc)
            : _owner(getTransceiver()->rank()),
              _slice(std::forward<PVSlice>(slc)),
              _lslice(to_xt(_slice.tile_slice())),
              _xarray()
        {
            _xarray = org;
        }

        ~DPTensorX()
        {
        }

        bool is_sliced() const
        {
            return _slice.is_sliced();
        }

        virtual std::string __repr__() const
        {
            auto v = xt::strided_view(xarray(), lslice());
            std::ostringstream oss;
            oss << v << "\n";
            return oss.str();
        }

        virtual DTypeId dtype() const
        {
            return DTYPE<T>::value;
        }

        virtual const shape_type & shape() const
        {
            return _slice.shape();
        }

        virtual int ndim() const
        {
            return _slice.ndims();
        }

        virtual uint64_t size() const
        {
            return _slice.size();
        }

        friend struct Service;

        virtual bool __bool__() const
        {
            if(! is_replicated())
                throw(std::runtime_error("Cast to scalar bool: tensor is not replicated"));
            return static_cast<bool>(_replica);
        }

        virtual double __float__() const
        {
            if(! is_replicated())
                throw(std::runtime_error("Cast to scalar float: tensor is not replicated"));
            return static_cast<double>(_replica);
        }

        virtual int64_t __int__() const
        {
            if(! is_replicated())
                throw(std::runtime_error("Cast to scalar int: tensor is not replicated"));
            return static_cast<int64_t>(_replica);
        }

        virtual uint64_t __len__() const
        {
            return _slice.slice().dim(0).size();
        }

        xt::xarray<T> & xarray()
        {
            return *_xarray.get();
        }

        const xt::xarray<T> & xarray() const
        {
            return *_xarray.get();
        }

        const PVSlice & slice() const
        {
            return _slice;
        }

        const xt::xstrided_slice_vector & lslice() const
        {
            return _lslice;
        }

        bool has_owner() const
        {
            return _owner < _OWNER_END;
        }

        void set_owner(rank_type o) const
        {
            _owner = o;
        }

        rank_type owner() const
        {
            return _owner;
        }

        bool is_replicated() const
        {
            return _owner == REPLICATED;
        }

        virtual int item_size() const
        {
            return sizeof(T);
        }

        virtual void bufferize(const NDSlice & slc, Buffer & buff) const
        {
            if(slc.size() <= 0) return;
            NDSlice lslice = NDSlice(slice().tile_shape()).slice(slc);

            auto ary_v = xt::strided_view(xarray(), to_xt(lslice));
            auto pos = buff.size();
            buff.resize(pos + lslice.size()*sizeof(T));
            T * out = reinterpret_cast<T*>(buff.data() + pos);
            pos = 0;
            for(auto i = ary_v.begin(); i != ary_v.end(); ++i, ++pos) {
                out[pos] = *i;
            }
        }
    };

    template<typename T>
    class operatorx
    {
    public:
        template<typename ...Ts>
        static typename DPTensorX<T>::typed_ptr_type mk_tx(Ts&&... args)
        {
            return std::make_shared<DPTensorX<T>>(std::forward<Ts>(args)...);
        }

        static typename DPTensorX<T>::typed_ptr_type mk_tx(
            uint64_t rank,
            void *allocated,
            void *aligned,
            intptr_t offset,
            const intptr_t * sizes,
            const intptr_t * strides)
        {
            // FIXME strides/slices are not used
            T * dptr = reinterpret_cast<T*>(aligned) + offset;
            if(rank == 0) {
                return std::make_shared<DPTensorX<T>>(*dptr);
            }
            shape_type shp(sizes, sizes+rank);
            return std::make_shared<DPTensorX<T>>(shp, dptr);
        }

        template<typename X>
        static DPTensorBaseX::ptr_type mk_tx_(const DPTensorX<T> & tx, X && x)
        {
            return std::make_shared<DPTensorX<typename X::value_type>>(tx.shape(), std::forward<X>(x));
        }

        template<typename X>
        static DPTensorBaseX::ptr_type mk_tx_(const typename DPTensorX<T>::typed_ptr_type & tx, X && x)
        {
            return std::make_shared<DPTensorX<typename X::value_type>>(tx->shape(), std::forward<X>(x));
        }

        template<typename ...Ts>
        static tensor_i::future_type mk_ftx(Ts&&... args)
        {
            return UnDeferred(operatorx<T>::mk_tx(std::forward(args)...)).get_future();
        }
    };

} // namespace x
