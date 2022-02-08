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
            auto _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
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
} // namespace x
