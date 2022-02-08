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
            case IMOD:
                a %= b;
                return;
            case IOR:
                a |= b;
                return;
            case IAND:
                a &= b;
                return;
            case ILSHIFT:
                a = xt::left_shift(a, b);
                return;
            case IRSHIFT:
                a = xt::right_shift(a, b);
                return;
            case IXOR:
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
            case IADD:
                a += b;
                return;
            case IFLOORDIV:
                a = xt::floor(a / b);
                return;
            case IMUL:
                a *= b;
                return;
                /* FIXME
            case IPOW:
                a = xt::pow(a, b);
                return;
                */
            case ISUB:
                a -= b;
                return;
            case ITRUEDIV:
                a /= b;
                return;
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

        template<typename X>
        static ptr_type mk_tx(const DPTensorBaseX & tx, X && x)
        {
            return std::make_shared<DPTensorX<typename X::value_type>>(tx.shape(), x);
        }

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
            case AND:
            case RAND:
                return mk_tx(tx, a & b);
            case LSHIFT:
                return mk_tx(tx, a << b);
            case MOD:
                return mk_tx(tx, a % b);
            case OR:
            case ROR:
                return mk_tx(tx, a | b);
            case RSHIFT:
                return mk_tx(tx, a >> b);
            case XOR:
            case RXOR:
                return mk_tx(tx, a ^ b);
            case RLSHIFT:
                return mk_tx(tx, b << a);
            case RMOD:
                return mk_tx(tx, b % a);
            case RRSHIFT:
                return mk_tx(tx, b >> a);
            default:
                throw std::runtime_error("Unknown elementwise binary operation");
            }
        }

        static ptr_type op(EWBinOpId bop, const ptr_type & a_ptr, const ptr_type & b_ptr)
        {
            auto _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            auto const _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto & a = _a->xarray();
            auto const & b = _b->xarray();
            
            switch(bop) {
            case ADD:
            case RADD:
                return mk_tx(*_a, a + b);
            case EQ:
                return  mk_tx(*_a, xt::equal(a, b));
            case FLOORDIV:
                return mk_tx(*_a, xt::floor(a / b));
            case GE:
                return mk_tx(*_a, a >= b);
            case GT:
                return mk_tx(*_a, a > b);
            case LE:
                return mk_tx(*_a, a <= b);
            case LT:
                return mk_tx(*_a, a < b);
                /* FIXME
            case MATMUL:
                return mk_tx(*_a, );
                */
            case MUL:
            case RMUL:
                return mk_tx(*_a, a * b);
            case NE:
                return mk_tx(*_a, xt::not_equal(a, b));
                /* FIXME
            case POW:
                return mk_tx(*_a, );
                */
            case SUB:
                return mk_tx(*_a, a - b);
            case TRUEDIV:
                return mk_tx(*_a, a / b);
            case RFLOORDIV:
                return mk_tx(*_a, xt::floor(b / a));
                /* FIXME
            case RPOW:
                return mk_tx(*_a, );
                */
            case RSUB:
                return mk_tx(*_a, b - a);
            case RTRUEDIV:
                return mk_tx(*_a, b / a);
            }
            return integral_op(bop, *_a, a, b);
        }

#pragma GCC diagnostic pop

    };
} // namespace x
