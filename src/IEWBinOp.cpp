#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"

namespace x {

    class IEWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename A, typename B>
        static void op(IEWBinOpId iop, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            auto & ax = a_ptr->xarray();
            const auto & bx = b_ptr->xarray();
            if(a_ptr->is_sliced() || b_ptr->is_sliced()) {
                auto av = xt::strided_view(ax, a_ptr->lslice());
                const auto & bv = xt::strided_view(bx, b_ptr->lslice());
                do_op(iop, av, bv);
            } else {
                do_op(iop, ax, bx);
            }
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename T1, typename T2>
        static void do_op(IEWBinOpId iop, T1 & a, const T2 & b)
        {
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
            if constexpr (std::is_integral<typename T1::value_type>::value && std::is_integral<typename T2::value_type>::value) {
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

void IEWBinOp::op(IEWBinOpId op, x::DPTensorBaseX::ptr_type a, py::object & b)
{
    auto bb = x::mk_tx(b);
    TypeDispatch<x::IEWBinOp>(a, bb, op);
}
