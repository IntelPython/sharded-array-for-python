#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

    class IEWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename A, typename B>
        static void op(IEWBinOpId iop, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            auto a = xt::strided_view(a_ptr->xarray(), a_ptr->lslice());
            const auto b = xt::strided_view(b_ptr->xarray(), b_ptr->lslice());
            
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
            if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
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
                case __IXOR__:
                    a ^= b;
                case __ILSHIFT__:
                    a = xt::left_shift(a, b);
                    return;
                case __IRSHIFT__:
                    a = xt::right_shift(a, b);
                    return;
                }
            }
            throw std::runtime_error("Unknown/invalid inplace elementwise binary operation");
        }
#pragma GCC diagnostic pop

    };
} // namespace x

void IEWBinOp::op(IEWBinOpId op, x::DPTensorBaseX::ptr_type a, x::DPTensorBaseX::ptr_type b)
{
    TypeDispatch<x::IEWBinOp>(a, b, op);
}
