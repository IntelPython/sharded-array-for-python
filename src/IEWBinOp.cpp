#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"

namespace x {

    class IEWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename A, typename B>
        static ptr_type op(IEWBinOpId iop, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            auto & ax = a_ptr->xarray();
            const auto & bx = b_ptr->xarray();
            if(a_ptr->is_sliced() || b_ptr->is_sliced()) {
                auto av = xt::strided_view(ax, a_ptr->lslice());
                const auto & bv = xt::strided_view(bx, b_ptr->lslice());
                return do_op(iop, av, bv, a_ptr);
            }
            return do_op(iop, ax, bx, a_ptr);
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename A, typename T1, typename T2>
        static ptr_type do_op(IEWBinOpId iop, T1 & a, const T2 & b, std::shared_ptr<DPTensorX<A>> a_ptr)
        {
            switch(iop) {
            case __IADD__:
                a += b;
                return a_ptr;
            case __IFLOORDIV__:
                a = xt::floor(a / b);
                return a_ptr;
            case __IMUL__:
                a *= b;
                return a_ptr;
            case __ISUB__:
                a -= b;
                return a_ptr;
            case __ITRUEDIV__:
                a /= b;
                return a_ptr;
            case __IPOW__:
                throw std::runtime_error("Binary inplace operation not implemented");
            }
            if constexpr (std::is_integral<typename T1::value_type>::value && std::is_integral<typename T2::value_type>::value) {
                switch(iop) {
                case __IMOD__:
                    a %= b;
                    return a_ptr;
                case __IOR__:
                    a |= b;
                    return a_ptr;
                case __IAND__:
                    a &= b;
                    return a_ptr;
                case __IXOR__:
                    a ^= b;
                case __ILSHIFT__:
                    a = xt::left_shift(a, b);
                    return a_ptr;
                case __IRSHIFT__:
                    a = xt::right_shift(a, b);
                    return a_ptr;
                }
            }
            throw std::runtime_error("Unknown/invalid inplace elementwise binary operation");
        }
#pragma GCC diagnostic pop

    };
} // namespace x

tensor_i::future_type IEWBinOp::op(IEWBinOpId op, tensor_i::future_type & a, const py::object & b)
{
    auto bb = x::mk_ftx(b);
    auto aa = std::move(a.get());
    auto bbb = std::move(bb.get());
    return defer([op, aa, bbb](){
            return TypeDispatch<x::IEWBinOp>(aa, bbb, op);
        });
}
