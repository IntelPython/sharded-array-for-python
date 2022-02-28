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
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xio.hpp>
#include <pybind11/numpy.h>
#include "Mediator.hpp"
#include "PVSlice.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"

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
        uint64_t _id = 0;
        mutable rank_type _owner;
        PVSlice _slice;
        xt::xstrided_slice_vector _lslice;
        std::shared_ptr<xt::xarray<T>> _xarray;
        mutable T _replica = 0;

    public:
        using typed_ptr_type = std::shared_ptr<DPTensorX<T>>;

        template<typename I>
        DPTensorX(PVSlice && slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(std::move(slc)),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
        }

        template<typename I>
        DPTensorX(const shape_type & slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(slc),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
        }

        template<typename I>
        DPTensorX(shape_type && slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(std::move(slc)),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
        }

        DPTensorX(const shape_type & shp, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(shp),
              _xarray(std::make_shared<xt::xarray<T>>(xt::empty<T>(_slice.shape_of_rank())))
        {
        }

        DPTensorX(const T & v, rank_type owner=theTransceiver->rank())
            : _owner(owner),
              _slice(shape_type{1}),
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
              _lslice(to_xt(_slice.local_slice_of_rank())),
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
            : _owner(theTransceiver->rank()),
              _slice(std::forward<PVSlice>(slc)),
              _lslice(to_xt(_slice.slice())),
              _xarray()
        {
            _xarray = org;
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

        virtual shape_type shape() const
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

        virtual bool __bool__() const
        {
            return static_cast<bool>(replicate());
        }

        virtual double __float__() const
        {
            return static_cast<double>(replicate());
        }

        virtual int64_t __int__() const
        {
            return static_cast<int64_t>(replicate());
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

        T replicate() const
        {
            std::cerr << "is_replicated()=" << is_replicated() << " owner=" << owner() << " shape=" << to_string(shape()) << std::endl;
            if(is_replicated()) return _replica;
            if(has_owner() && _slice.size() == 1) {
                if(theTransceiver->rank() == owner()) {
                    _replica = *(xt::strided_view(xarray(), lslice()).begin());
                    std::cerr << "replica: " << _replica << std::endl;
                }
                theTransceiver->bcast(&_replica, sizeof(T), owner());
                set_owner(REPLICATED);
            } else {
                throw(std::runtime_error("Replication implemented for single element and single owner only."));
            }
            return _replica;
        }

        virtual int item_size() const
        {
            return sizeof(T);
        }

        virtual uint64_t id() const
        {
            return _id;
        }

        void set_id(uint64_t id)
        {
            _id = id;
        }

        virtual void bufferize(const NDSlice & slc, Buffer & buff) const
        {
            NDSlice lslice = NDSlice(slice().shape_of_rank()).slice(slc);

            std::cerr << "lslice=" << lslice << " slc= " << slc << " buffsz=" << buff.size() << " want " << slc.size()*sizeof(T) << std::endl;

            auto ary_v = xt::strided_view(xarray(), to_xt(lslice));
            buff.resize(slc.size()*sizeof(T));
            T * out = reinterpret_cast<T*>(buff.data());
            int o = 0;
            for(auto i = ary_v.begin(); i != ary_v.end(); ++i, ++o) {
                out[o] = *i;
            }
        }
    };


    template<typename T>
    static typename DPTensorX<T>::typed_ptr_type register_tensor(typename DPTensorX<T>::typed_ptr_type t)
    {
        auto id = theMediator->register_array(t);
        t->set_id(id);
        return t;
    }

    template<typename T>
    class operatorx
    {
    public:

        static DPTensorBaseX::ptr_type mk_tx(py::object & o)
        {
            return std::make_shared<DPTensorX<T>>(o.cast<T>());
        }

        template<typename ...Ts>
        static typename DPTensorX<T>::typed_ptr_type mk_tx(Ts&&... args)
        {
            return register_tensor<T>(std::make_shared<DPTensorX<T>>(std::forward<Ts>(args)...));
        }

        template<typename X>
        static DPTensorBaseX::ptr_type mk_tx_(const DPTensorX<T> & tx, X && x)
        {
            return register_tensor<typename X::value_type>(std::make_shared<DPTensorX<typename X::value_type>>(tx.shape(), std::forward<X>(x)));
        }

        template<typename X>
        static DPTensorBaseX::ptr_type mk_tx_(const typename DPTensorX<T>::typed_ptr_type & tx, X && x)
        {
            return register_tensor<typename X::value_type>(std::make_shared<DPTensorX<typename X::value_type>>(tx->shape(), std::forward<X>(x)));
        }
    };

    static DPTensorBaseX::ptr_type mk_tx(py::object & b)
    {
        if(py::isinstance<DPTensorBaseX>(b)) {
            return b.cast<DPTensorBaseX::ptr_type>();
        } else if(py::isinstance<py::float_>(b)) {
            return operatorx<double>::mk_tx(b);
        } else if(py::isinstance<py::int_>(b)) {
            return operatorx<int64_t>::mk_tx(b);
        }
        throw std::runtime_error("Invalid right operand to elementwise binary operation");
    };

} // namespace x
