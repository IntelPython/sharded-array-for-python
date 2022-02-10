// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>
#include <sstream>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xio.hpp>
#include "PVSlice.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"

namespace x
{
#if 0
    template<typename T, typename S>
    T to_native(S v)
    {
        return static_cast<T>(v);
    }
#endif
    template<typename T>
    T to_native(py::object & o)
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
        rank_type _owner;
        PVSlice _slice;
        xt::xstrided_slice_vector _lslice;
        std::shared_ptr<xt::xarray<T>> _xarray;
        T _replica = 0;

    public:
        template<typename I>
        DPTensorX(PVSlice && slc, I && ax, rank_type owner=NOOWNER)
            : _owner(owner),
              _slice(std::move(slc)),
              _lslice(to_xt(_slice.local_slice_of_rank())),
              _xarray(std::make_shared<xt::xarray<T>>(std::forward<I>(ax)))
        {
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
            std::cerr << "slice: " << _slice.slice() << " sz " << _slice.size()
                      << " lslice: " << _slice.local_slice_of_rank() << " owner: " << _owner
                      << " val: " << _replica << std::endl;
        }

        virtual std::string __repr__() const
        {
            auto v = xt::strided_view(xarray(), lslice());
            std::ostringstream oss;
            oss << v << "\n";
            return oss.str();
        }

        virtual DType dtype() const
        {
            return DTYPE<T>::value;
        }

        virtual shape_type shape() const
        {
            return _slice.shape();
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
        
        void set_owner(rank_type o)
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

        T replicate()
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

        virtual void bufferize(const NDSlice & slc, Buffer & buff)
        {
            NDSlice lslice = NDSlice(slice().tile_shape()).slice(slc);

            std::cerr << "lslice=" << lslice << " slc= " << slc << " buffsz=" << buff.size() << " want " << slc.size()*sizeof(T) << std::endl;

            auto ary_v = xt::strided_view(xarray(), to_xt(lslice));
            buff.resize(slc.size()*sizeof(T));
            T * out = reinterpret_cast<T*>(buff.data());
            int o = 0;
            for(auto i = ary_v.begin(); i != ary_v.end(); ++i, ++o) {
                std::cerr << o << " <- " << *i << std::endl;
                out[o] = *i;
            }
        }
    };


    template<typename T>
    class operatorx
    {
    public:
        static tensor_i::ptr_type register_tensor(std::shared_ptr<DPTensorX<T>> t)
        {
            auto id = theMediator->register_array(t);
            t->set_id(id);
            return t;
        }

        template<typename ...Ts>
        static DPTensorBaseX::ptr_type mk_tx(Ts&&... args)
        {
            return register_tensor(std::make_shared<DPTensorX<T>>(std::forward<Ts>(args)...));
        }
            
        template<typename X>
        static DPTensorBaseX::ptr_type mk_tx(const DPTensorBaseX & tx, X && x)
        {
            return register_tensor(std::make_shared<DPTensorX<typename X::value_type>>(tx.shape(), std::forward<X>(x)));
        }
    };

    template<typename T>
    class Creator
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        static ptr_type op(CreatorId c, shape_type && shp)
        {
            PVSlice pvslice(std::forward<shape_type>(shp));
            shape_type shape(std::move(pvslice.tile_shape()));
            switch(c) {
            case EMPTY:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::empty<T>(shape)));
            case ONES:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::ones<T>(shape)));
            case ZEROS:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::zeros<T>(shape)));
            default:
                throw std::runtime_error("Unknown creator");
            };
        };

        template<typename V>
        static ptr_type op(CreatorId c, shape_type && shp, V && v)
        {
            if(c == FULL) {
                PVSlice pvslice(std::forward<shape_type>(shp));
                shape_type shape(std::move(pvslice.tile_shape()));
                auto a = xt::empty<T>(std::move(shape));
                a.fill(to_native<T>(v));
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
            }
            throw std::runtime_error("Unknown creator");
        }
    }; // class creatorx

    template<typename T>
    class IEWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"

        template<typename A, typename B, typename U = T, std::enable_if_t<std::is_floating_point<U>::value, bool> = true>
        static void integral_iop(IEWBinOpId iop, A && a, B && b)
        {
            throw std::runtime_error("Illegal or unknown inplace elementwise binary operation");
        }

        template<typename A, typename B, typename U = T, std::enable_if_t<std::is_integral<U>::value, bool> = true>
        static void integral_iop(IEWBinOpId iop, A && a, B && b)
        {
            switch(iop) {
            case __IMOD__:
                a %= b;
                return;
            case __IOR__:
                a |= b;
                return;
            case __IAND__:
                a &= b;
                return;
            case __ILSHIFT__:
                a = xt::left_shift(a, b);
                return;
            case __IRSHIFT__:
                a = xt::right_shift(a, b);
                return;
            case __IXOR__:
                a ^= b;
                return;
            default:
                throw std::runtime_error("Unknown inplace elementwise binary operation");
            }
        }

        static void op(IEWBinOpId iop, ptr_type a_ptr, const ptr_type & b_ptr)
        {
            auto _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            auto const _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto a = xt::strided_view(_a->xarray(), _a->lslice());
            auto const b = xt::strided_view(_b->xarray(), _b->lslice());
            
            switch(iop) {
            case __IADD__:
                a += b;
                return;
            case __IFLOORDIV__:
                a = xt::floor(a / b);
                return;
            case __IMUL__:
                a *= b;
                return;
            case __ISUB__:
                a -= b;
                return;
            case __ITRUEDIV__:
                a /= b;
                return;
            case __IPOW__:
                throw std::runtime_error("Binary inplace operation not implemented");
            }
            integral_iop(iop, a, b);
        }

#pragma GCC diagnostic pop

    };

    template<typename T>
    class EWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"

        template<typename A, typename B, typename U = T, std::enable_if_t<std::is_floating_point<U>::value, bool> = true>
        static ptr_type integral_op(EWBinOpId iop, const DPTensorX<T> & tx, A && a, B && b)
        {
            throw std::runtime_error("Illegal or unknown inplace elementwise binary operation");
        }

        template<typename A, typename B, typename U = T, std::enable_if_t<std::is_integral<U>::value, bool> = true>
        static ptr_type integral_op(EWBinOpId iop, const DPTensorBaseX & tx, A && a, B && b)
        {
            switch(iop) {
            case __AND__:
            case BITWISE_AND:
                return operatorx<T>::mk_tx(tx, a & b);
            case __RAND__:
                return operatorx<T>::mk_tx(tx, b & a);
            case __LSHIFT__:
            case BITWISE_LEFT_SHIFT:
                return operatorx<T>::mk_tx(tx, a << b);
            case __MOD__:
            case REMAINDER:
                return operatorx<T>::mk_tx(tx, a % b);
            case __OR__:
            case BITWISE_OR:
                return operatorx<T>::mk_tx(tx, a | b);
            case __ROR__:
                return operatorx<T>::mk_tx(tx, b | a);
            case __RSHIFT__:
            case BITWISE_RIGHT_SHIFT:
                return operatorx<T>::mk_tx(tx, a >> b);
            case __XOR__:
            case BITWISE_XOR:
                return operatorx<T>::mk_tx(tx, a ^ b);
            case __RXOR__:
                return operatorx<T>::mk_tx(tx, b ^ a);
            case __RLSHIFT__:
                return operatorx<T>::mk_tx(tx, b << a);
            case __RMOD__:
                return operatorx<T>::mk_tx(tx, b % a);
            case __RRSHIFT__:
                return operatorx<T>::mk_tx(tx, b >> a);
            default:
                throw std::runtime_error("Unknown elementwise binary operation");
            }
        }

        static ptr_type op(EWBinOpId bop, const ptr_type & a_ptr, const ptr_type & b_ptr)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            auto const _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto const & a = xt::strided_view(_a->xarray(), _a->lslice());
            auto const & b = xt::strided_view(_b->xarray(), _b->lslice());
            
            switch(bop) {
            case __ADD__:
            case ADD:
                return operatorx<T>::mk_tx(*_a, a + b);
            case __RADD__:
                return operatorx<T>::mk_tx(*_a, b + a);
            case ATAN2:
                return  operatorx<T>::mk_tx(*_a, xt::atan2(a, b));
            case __EQ__:
            case EQUAL:
                return  operatorx<T>::mk_tx(*_a, xt::equal(a, b));
            case __FLOORDIV__:
            case FLOOR_DIVIDE:
                return operatorx<T>::mk_tx(*_a, xt::floor(a / b));
            case __GE__:
            case GREATER_EQUAL:
                return operatorx<T>::mk_tx(*_a, a >= b);
            case __GT__:
            case GREATER:
                return operatorx<T>::mk_tx(*_a, a > b);
            case __LE__:
            case LESS_EQUAL:
                return operatorx<T>::mk_tx(*_a, a <= b);
            case __LT__:
            case LESS:
                return operatorx<T>::mk_tx(*_a, a < b);
            case __MUL__:
            case MULTIPLY:
                return operatorx<T>::mk_tx(*_a, a * b);
            case __RMUL__:
                return operatorx<T>::mk_tx(*_a, b * a);
            case __NE__:
            case NOT_EQUAL:
                return operatorx<T>::mk_tx(*_a, xt::not_equal(a, b));
            case __SUB__:
            case SUBTRACT:
                return operatorx<T>::mk_tx(*_a, a - b);
            case __TRUEDIV__:
            case DIVIDE:
                return operatorx<T>::mk_tx(*_a, a / b);
            case __RFLOORDIV__:
                return operatorx<T>::mk_tx(*_a, xt::floor(b / a));
            case __RSUB__:
                return operatorx<T>::mk_tx(*_a, b - a);
            case __RTRUEDIV__:
                return operatorx<T>::mk_tx(*_a, b / a);
            case __MATMUL__:
            case __POW__:
            case POW:
            case __RPOW__:
            case LOGADDEXP:
            case LOGICAL_AND:
            case LOGICAL_OR:
            case LOGICAL_XOR:
                // FIXME
                throw std::runtime_error("Binary operation not implemented");
            }
            return integral_op(bop, *_a, a, b);
        }

#pragma GCC diagnostic pop

    };

    
    template<typename T>
    class EWUnyOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"

        template<typename A, typename U = T, std::enable_if_t<std::is_floating_point<U>::value, bool> = true>
        static ptr_type integral_op(EWUnyOpId uop, const DPTensorBaseX & tx, A && a)
        {
            throw std::runtime_error("Illegal or unknown inplace elementwise unary operation");
        }

        template<typename A, typename U = T, std::enable_if_t<std::is_integral<U>::value, bool> = true>
        static ptr_type integral_op(EWUnyOpId uop, const DPTensorBaseX & tx, A && a)
        {
            switch(uop) {
            case __INVERT__:
            case BITWISE_INVERT:
            default:
                throw std::runtime_error("Unknown elementwise unary operation");
            }
        }

        static ptr_type op(EWUnyOpId uop, const ptr_type & a_ptr)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto const & a = xt::strided_view(_a->xarray(), _a->lslice());
            
            switch(uop) {
            case __ABS__:
            case ABS:
                return operatorx<T>::mk_tx(*_a, xt::abs(a));
            case ACOS:
                return operatorx<T>::mk_tx(*_a, xt::acos(a));
            case ACOSH:
                return operatorx<T>::mk_tx(*_a, xt::acosh(a));
            case ASIN:
                return operatorx<T>::mk_tx(*_a, xt::asin(a));
            case ASINH:
                return operatorx<T>::mk_tx(*_a, xt::asinh(a));
            case ATAN:
                return operatorx<T>::mk_tx(*_a, xt::atan(a));
            case ATANH:
                return operatorx<T>::mk_tx(*_a, xt::atanh(a));
            case CEIL:
                return operatorx<T>::mk_tx(*_a, xt::ceil(a));
            case COS:
                return operatorx<T>::mk_tx(*_a, xt::cos(a));
            case COSH:
                return operatorx<T>::mk_tx(*_a, xt::cosh(a));
            case EXP:
                return operatorx<T>::mk_tx(*_a, xt::exp(a));
            case EXPM1:
                return operatorx<T>::mk_tx(*_a, xt::expm1(a));
            case FLOOR:
                return operatorx<T>::mk_tx(*_a, xt::floor(a));
            case ISFINITE:
                return operatorx<T>::mk_tx(*_a, xt::isfinite(a));
            case ISINF:
                return operatorx<T>::mk_tx(*_a, xt::isinf(a));
            case ISNAN:
                return operatorx<T>::mk_tx(*_a, xt::isnan(a));
            case LOG:
                return operatorx<T>::mk_tx(*_a, xt::log(a));
            case LOG1P:
                return operatorx<T>::mk_tx(*_a, xt::log1p(a));
            case LOG2:
                return operatorx<T>::mk_tx(*_a, xt::log2(a));
            case LOG10:
                return operatorx<T>::mk_tx(*_a, xt::log10(a));
            case ROUND:
                return operatorx<T>::mk_tx(*_a, xt::round(a));
            case SIGN:
                return operatorx<T>::mk_tx(*_a, xt::sign(a));
            case SIN:
                return operatorx<T>::mk_tx(*_a, xt::sin(a));
            case SINH:
                return operatorx<T>::mk_tx(*_a, xt::sinh(a));
            case SQUARE:
                return operatorx<T>::mk_tx(*_a, xt::square(a));
            case SQRT:
                return operatorx<T>::mk_tx(*_a, xt::sqrt(a));
            case TAN:
                return operatorx<T>::mk_tx(*_a, xt::tan(a));
            case TANH:
                return operatorx<T>::mk_tx(*_a, xt::tanh(a));
            case TRUNC:
                return operatorx<T>::mk_tx(*_a, xt::trunc(a));
            case __NEG__:
            case NEGATIVE:
            case __POS__:
            case POSITIVE:
            case LOGICAL_NOT:
                // FIXME
                throw std::runtime_error("Unary operation not implemented");
            }
            return integral_op(uop, *_a, a);
        }

#pragma GCC diagnostic pop

    };

    template<typename T>
    class UnyOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename N>
        static N __type__(const ptr_type & a_ptr)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            T v = _a->replicate();
            return static_cast<N>(v);
        }
        static bool op(const ptr_type & a_ptr, bool)
        {
            return __type__<bool>(a_ptr);
        }
        static double op(const ptr_type & a_ptr, double)
        {
            return __type__<double>(a_ptr);
        }
        static int64_t op(const ptr_type & a_ptr, int64_t)
        {
            return __type__<int64_t>(a_ptr);
        }
    };
    
    template<typename T>
    class ReduceOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"

        template<typename X>
        static ptr_type dist_reduce(ReduceOpId rop, const PVSlice & slice, const dim_vec_type & dims, X && x)
        {
            xt::xarray<typename X::value_type> a = x;
            auto new_shape = reduce_shape(slice.shape(), dims);
            rank_type owner = NOOWNER;
            if(slice.need_reduce(dims)) {
                auto len = VPROD(new_shape);
                theTransceiver->reduce_all(a.data(), DTYPE<typename X::value_type>::value, len, rop);
                owner = REPLICATED;
            }
            return operatorx<typename X::value_type>::mk_tx(new_shape, a, owner);
        }

        static ptr_type op(ReduceOpId rop, const ptr_type & a_ptr, const dim_vec_type & dims)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto const & a = xt::strided_view(_a->xarray(), _a->lslice());

            switch(rop) {
            case MEAN:
                return dist_reduce(rop, _a->slice(), dims, xt::mean(a, dims));
            case PROD:
                return dist_reduce(rop, _a->slice(), dims, xt::prod(a, dims));
            case SUM:
                return dist_reduce(rop, _a->slice(), dims, xt::sum(a, dims));
            case STD:
                return dist_reduce(rop, _a->slice(), dims, xt::stddev(a, dims));
            case VAR:
                return dist_reduce(rop, _a->slice(), dims, xt::variance(a, dims));
            case MAX:
            case MIN:
                throw std::runtime_error("Reduction operation not implemented");
            default:
                throw std::runtime_error("Unknown reduction operation");
            }
        }

#pragma GCC diagnostic pop

    };

    template<typename T>
    class GetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        static ptr_type op(const ptr_type & a_ptr, const NDSlice & slice)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto nd = _a->shape().size();
            if(nd != slice.ndims())
                throw std::runtime_error("Index dimensionality must match array dimensionality");

            return operatorx<T>::mk_tx(*_a, slice);
        }
    };

    template<typename T>
    class SetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // copy data from val into (*dest)[slice]
        // this is a non-collective call.
        template<typename U>
        static void _set_slice(DPTensorX<T> & dest, const NDSlice & dest_slice, const DPTensorX<U> & val, const NDSlice & val_slice)
        {
            std::cerr << "_set_slice " << dest.slice() << " " << dest_slice << " " << val.slice() << " " << val_slice << std::endl;
            auto nd = dest.shape().size();
            if(dest.owner() == REPLICATED && nd > 0)
                std::cerr << "Warning: __setitem__ on replicated data updates local tile only" << std::endl;
            if(nd != dest_slice.ndims())
                throw std::runtime_error("Index dimensionality must match array dimensionality");
            if(val_slice.size() != dest_slice.size())
                throw std::runtime_error("Input and output slices must be of same size");

            // Use given slice to create a global view into orig array
            PVSlice g_slc_view(dest.slice(), dest_slice);
            std::cerr << "g_slice: " << g_slc_view.slice() << std::endl;
            // Create a view into val
            PVSlice needed_val_view(val.slice(), val_slice);
            std::cerr << "needed_val_view: " << needed_val_view.slice() << " (was " << val.slice().slice() << ")" << std::endl;

            // we can now compute which ranks actually hold which piece of the data from val that we need locally
            for(rank_type i=0; i<theTransceiver->nranks(); ++i ) {
                // get local view into val
                PVSlice val_local_view(val.slice(), i);
                std::cerr << i << " val_local_view: " << val_local_view.slice() << std::endl;
                NDSlice curr_needed_val_slice = needed_val_view.slice_of_rank(i);
                std::cerr << i << " curr_needed_val_slice: " << curr_needed_val_slice << std::endl;
                NDSlice curr_local_val_slice = val_local_view.map_slice(curr_needed_val_slice);
                std::cerr << i << " curr_local_val_slice: " << curr_local_val_slice << std::endl;
                NDSlice curr_needed_norm_slice = needed_val_view.map_slice(curr_needed_val_slice);
                std::cerr << i << " curr_needed_norm_slice: " << curr_needed_norm_slice << std::endl;
                PVSlice my_curr_needed_view = PVSlice(g_slc_view, curr_needed_norm_slice);
                std::cerr << i << " my_curr_needed_slice: " << my_curr_needed_view.slice() << std::endl;
                NDSlice my_curr_local_slice = my_curr_needed_view.local_slice_of_rank(theTransceiver->rank());
                std::cerr << i << " my_curr_local_slice: " << my_curr_local_slice << std::endl;
                if(curr_needed_norm_slice.size()) {
                    py::tuple tpl = _make_tuple(my_curr_local_slice); //my_curr_view.slice());
                    if(i == theTransceiver->rank()) {
                        // copy locally
                        auto to_v   = xt::strided_view(dest.xarray(), to_xt(my_curr_local_slice));
                        auto from_v = xt::strided_view(val.xarray(), to_xt(curr_local_val_slice));
                        to_v = from_v;
                    } else {
                        // pull slice directly into new array
                        xt::xarray<U> from_a = xt::empty<U>(curr_local_val_slice.shape());
                        from_a.fill(static_cast<U>(4711));
                        theMediator->pull(i, val, curr_local_val_slice, from_a.data());
                        auto to_v = xt::strided_view(dest.xarray(), to_xt(my_curr_local_slice));
                        to_v = from_a;
                    }
                }
            }
        }

        // FIXME We use a generic SPMD/PGAS mechanism to pull elements from remote
        // on all procs simultaneously.  Since __setitem__ is collective we could
        // implement a probaly more efficient mechanism which pushes data and/or using RMA.
        static void op(ptr_type & a_ptr, const NDSlice & slice, const ptr_type & b_ptr)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            auto const _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");

            // Use given slice to create a global view into orig array
            PVSlice g_slc_view(_a->slice(), slice);
            std::cerr << "g_slice: " << g_slc_view.slice() << std::endl;
            NDSlice my_slice = g_slc_view.slice_of_rank();
            std::cerr << "my_slice: " << my_slice << std::endl;
            NDSlice my_norm_slice = g_slc_view.map_slice(my_slice);
            std::cerr << "my_norm_slice: " << my_norm_slice << std::endl;
            NDSlice my_rel_slice = _a->slice().map_slice(my_slice);
            std::cerr << "my_rel_slice: " << my_rel_slice << std::endl;
            
            theTransceiver->barrier();
            _set_slice(*_a, my_rel_slice, *_b, my_norm_slice);
        }

    };
} // namespace x
