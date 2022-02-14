#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

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
            const auto _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            auto a = xt::strided_view(_a->xarray(), _a->lslice());
            const auto b = xt::strided_view(_b->xarray(), _b->lslice());
            
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
} // namespace x

void IEWBinOp::op(IEWBinOpId op, x::DPTensorBaseX::ptr_type a, x::DPTensorBaseX::ptr_type b)
{
    TypeDispatch<x::IEWBinOp>(a->dtype(), op, a, b);
}
