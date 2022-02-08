// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>
#include <sstream>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include "PVSlice.hpp"
#include "p2c_ids.hpp"

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

    class DPTensorBaseX
    {
    public:
        using ptr_type = std::shared_ptr<DPTensorBaseX>;

        virtual ~DPTensorBaseX() {};
        virtual std::string __repr__() const = 0;
        virtual DType dtype() const = 0;
        virtual shape_type shape() const = 0;
    };

    template<typename T>
    class DPTensorX : public DPTensorBaseX
    {
        PVSlice _slice;
        xt::xarray<T> _xarray;

    public:
        template<typename I>
        DPTensorX(PVSlice && slc, I && ax)
            : _slice(std::move(slc)),
              _xarray(std::forward<I>(ax))
        {
        }

        virtual std::string __repr__() const
        {
            std::ostringstream oss;
            oss << _xarray << "\n";
            return oss.str();
        }

        virtual DType dtype() const
        {
            return DTYPE<T>::value;
        }

        xt::xarray<T> & xarray()
        {
            return _xarray;
        }

        const PVSlice & slice() const
        {
            return _slice;
        }

        virtual shape_type shape() const
        {
            return _slice.shape();
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
                return std::make_shared<DPTensorX<T>>(std::move(pvslice), std::move(xt::empty<T>(shape)));
            case ONES:
                return std::make_shared<DPTensorX<T>>(std::move(pvslice), std::move(xt::ones<T>(shape)));
            case ZEROS:
                return std::make_shared<DPTensorX<T>>(std::move(pvslice), std::move(xt::zeros<T>(shape)));
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
                a = to_native<T>(v);
                return std::make_shared<DPTensorX<T>>(std::move(pvslice), std::move(a));
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
            auto & a = _a->xarray();
            auto const & b = _b->xarray();
            
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

    template<typename X>
    inline DPTensorBaseX::ptr_type mk_tx(const DPTensorBaseX & tx, X && x)
    {
        return std::make_shared<DPTensorX<typename X::value_type>>(tx.shape(), x);
    }
    
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
                return mk_tx(tx, a & b);
            case __RAND__:
                return mk_tx(tx, b & a);
            case __LSHIFT__:
            case BITWISE_LEFT_SHIFT:
                return mk_tx(tx, a << b);
            case __MOD__:
            case REMAINDER:
                return mk_tx(tx, a % b);
            case __OR__:
            case BITWISE_OR:
                return mk_tx(tx, a | b);
            case __ROR__:
                return mk_tx(tx, b | a);
            case __RSHIFT__:
            case BITWISE_RIGHT_SHIFT:
                return mk_tx(tx, a >> b);
            case __XOR__:
            case BITWISE_XOR:
                return mk_tx(tx, a ^ b);
            case __RXOR__:
                return mk_tx(tx, b ^ a);
            case __RLSHIFT__:
                return mk_tx(tx, b << a);
            case __RMOD__:
                return mk_tx(tx, b % a);
            case __RRSHIFT__:
                return mk_tx(tx, b >> a);
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
            auto const & a = _a->xarray();
            auto const & b = _b->xarray();
            
            switch(bop) {
            case __ADD__:
            case ADD:
                return mk_tx(*_a, a + b);
            case __RADD__:
                return mk_tx(*_a, b + a);
            case ATAN2:
                return  mk_tx(*_a, xt::atan2(a, b));
            case __EQ__:
            case EQUAL:
                return  mk_tx(*_a, xt::equal(a, b));
            case __FLOORDIV__:
            case FLOOR_DIVIDE:
                return mk_tx(*_a, xt::floor(a / b));
            case __GE__:
            case GREATER_EQUAL:
                return mk_tx(*_a, a >= b);
            case __GT__:
            case GREATER:
                return mk_tx(*_a, a > b);
            case __LE__:
            case LESS_EQUAL:
                return mk_tx(*_a, a <= b);
            case __LT__:
            case LESS:
                return mk_tx(*_a, a < b);
            case __MUL__:
            case MULTIPLY:
                return mk_tx(*_a, a * b);
            case __RMUL__:
                return mk_tx(*_a, b * a);
            case __NE__:
            case NOT_EQUAL:
                return mk_tx(*_a, xt::not_equal(a, b));
            case __SUB__:
            case SUBTRACT:
                return mk_tx(*_a, a - b);
            case __TRUEDIV__:
            case DIVIDE:
                return mk_tx(*_a, a / b);
            case __RFLOORDIV__:
                return mk_tx(*_a, xt::floor(b / a));
            case __RSUB__:
                return mk_tx(*_a, b - a);
            case __RTRUEDIV__:
                return mk_tx(*_a, b / a);
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
            auto const & a = _a->xarray();
            
            switch(uop) {
            case __ABS__:
            case ABS:
                return mk_tx(*_a, xt::abs(a));
            case ACOS:
                return mk_tx(*_a, xt::acos(a));
            case ACOSH:
                return mk_tx(*_a, xt::acosh(a));
            case ASIN:
                return mk_tx(*_a, xt::asin(a));
            case ASINH:
                return mk_tx(*_a, xt::asinh(a));
            case ATAN:
                return mk_tx(*_a, xt::atan(a));
            case ATANH:
                return mk_tx(*_a, xt::atanh(a));
            case CEIL:
                return mk_tx(*_a, xt::ceil(a));
            case COS:
                return mk_tx(*_a, xt::cos(a));
            case COSH:
                return mk_tx(*_a, xt::cosh(a));
            case EXP:
                return mk_tx(*_a, xt::exp(a));
            case EXPM1:
                return mk_tx(*_a, xt::expm1(a));
            case FLOOR:
                return mk_tx(*_a, xt::floor(a));
            case ISFINITE:
                return mk_tx(*_a, xt::isfinite(a));
            case ISINF:
                return mk_tx(*_a, xt::isinf(a));
            case ISNAN:
                return mk_tx(*_a, xt::isnan(a));
            case LOG:
                return mk_tx(*_a, xt::log(a));
            case LOG1P:
                return mk_tx(*_a, xt::log1p(a));
            case LOG2:
                return mk_tx(*_a, xt::log2(a));
            case LOG10:
                return mk_tx(*_a, xt::log10(a));
            case ROUND:
                return mk_tx(*_a, xt::round(a));
            case SIGN:
                return mk_tx(*_a, xt::sign(a));
            case SIN:
                return mk_tx(*_a, xt::sin(a));
            case SINH:
                return mk_tx(*_a, xt::sinh(a));
            case SQUARE:
                return mk_tx(*_a, xt::square(a));
            case SQRT:
                return mk_tx(*_a, xt::sqrt(a));
            case TAN:
                return mk_tx(*_a, xt::tan(a));
            case TANH:
                return mk_tx(*_a, xt::tanh(a));
            case TRUNC:
                return mk_tx(*_a, xt::trunc(a));
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
            if(slice.need_reduce(dims)) {
                auto len = VPROD(new_shape);
                theTransceiver->reduce_all(a.data(), DTYPE<typename X::value_type>::value, len, rop);
            }
            return std::make_shared<DPTensorX<typename X::value_type>>(new_shape, a);
        }

        static ptr_type op(ReduceOpId rop, const ptr_type & a_ptr, const dim_vec_type & dims)
        {
            auto const _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto const & a = _a->xarray();

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

                
} // namespace x
