#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"

namespace x {

    class EWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename A, typename B>
        static ptr_type op(EWBinOpId bop, const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            const auto & ax = a_ptr->xarray();
            const auto & bx = b_ptr->xarray();
            if(a_ptr->is_sliced() || b_ptr->is_sliced()) {
                const auto & av = xt::strided_view(ax, a_ptr->lslice());
                const auto & bv = xt::strided_view(bx, b_ptr->lslice());
                return do_op(bop, av, bv, a_ptr, b_ptr);
            }
            return do_op(bop, ax, bx, a_ptr, b_ptr);
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename T1, typename T2, typename A, typename B>
        static ptr_type do_op(EWBinOpId bop, const T1 & a, const T2 & b, const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            switch(bop) {
            case __ADD__:
            case ADD:
                return operatorx<A>::mk_tx_(a_ptr, a + b);
            case __RADD__:
                return operatorx<A>::mk_tx_(a_ptr, b + a);
            case ATAN2:
                return  operatorx<A>::mk_tx_(a_ptr, xt::atan2(a, b));
            case __EQ__:
            case EQUAL:
                return  operatorx<A>::mk_tx_(a_ptr, xt::equal(a, b));
            case __FLOORDIV__:
            case FLOOR_DIVIDE:
                return operatorx<A>::mk_tx_(a_ptr, xt::floor(a / b));
            case __GE__:
            case GREATER_EQUAL:
                return operatorx<A>::mk_tx_(a_ptr, a >= b);
            case __GT__:
            case GREATER:
                return operatorx<A>::mk_tx_(a_ptr, a > b);
            case __LE__:
            case LESS_EQUAL:
                return operatorx<A>::mk_tx_(a_ptr, a <= b);
            case __LT__:
            case LESS:
                return operatorx<A>::mk_tx_(a_ptr, a < b);
            // __MATMUL__ is handled before dispatching, see below
            case __MUL__:
            case MULTIPLY:
                return operatorx<A>::mk_tx_(a_ptr, a * b);
            case __RMUL__:
                return operatorx<A>::mk_tx_(a_ptr, b * a);
            case __NE__:
            case NOT_EQUAL:
                return operatorx<A>::mk_tx_(a_ptr, xt::not_equal(a, b));
            case __SUB__:
            case SUBTRACT:
                return operatorx<A>::mk_tx_(a_ptr, a - b);
            case __TRUEDIV__:
            case DIVIDE:
                return operatorx<A>::mk_tx_(a_ptr, a / b);
            case __RFLOORDIV__:
                return operatorx<A>::mk_tx_(a_ptr, xt::floor(b / a));
            case __RSUB__:
                return operatorx<A>::mk_tx_(a_ptr, b - a);
            case __RTRUEDIV__:
                return operatorx<A>::mk_tx_(a_ptr, b / a);
            case __POW__:
            case POW:
                return operatorx<A>::mk_tx_(a_ptr, xt::pow(a, b));
            case __RPOW__:
                return operatorx<A>::mk_tx_(a_ptr, xt::pow(b, a));
            case LOGADDEXP:
                return operatorx<A>::mk_tx_(a_ptr, xt::log(xt::exp(a) + xt::exp(b)));
            case LOGICAL_AND:
                return operatorx<A>::mk_tx_(a_ptr, xt::cast<bool>(a) && xt::cast<bool>(b));
            case LOGICAL_OR:
                return operatorx<A>::mk_tx_(a_ptr, xt::cast<bool>(a) || xt::cast<bool>(b));
            case LOGICAL_XOR:
                {
                    auto ba = xt::cast<bool>(a);
                    auto bb = xt::cast<bool>(b);
                    return operatorx<A>::mk_tx_(a_ptr, (b || a) && xt::not_equal(ba, bb));
                }
            }
            if constexpr (std::is_integral<A>::value && std::is_integral<typename T2::value_type>::value) {
                switch(bop) {
                case __AND__:
                case BITWISE_AND:
                    return operatorx<A>::mk_tx_(a_ptr, a & b);
                case __RAND__:
                    return operatorx<A>::mk_tx_(a_ptr, b & a);
                case __LSHIFT__:
                case BITWISE_LEFT_SHIFT:
                    return operatorx<A>::mk_tx_(a_ptr, a << b);
                case __MOD__:
                case REMAINDER:
                    return operatorx<A>::mk_tx_(a_ptr, a % b);
                case __OR__:
                case BITWISE_OR:
                    return operatorx<A>::mk_tx_(a_ptr, a | b);
                case __ROR__:
                    return operatorx<A>::mk_tx_(a_ptr, b | a);
                case __RSHIFT__:
                case BITWISE_RIGHT_SHIFT:
                    return operatorx<A>::mk_tx_(a_ptr, a >> b);
                case __XOR__:
                case BITWISE_XOR:
                    return operatorx<A>::mk_tx_(a_ptr, a ^ b);
                case __RXOR__:
                    return operatorx<A>::mk_tx_(a_ptr, b ^ a);
                case __RLSHIFT__:
                    return operatorx<A>::mk_tx_(a_ptr, b << a);
                case __RMOD__:
                    return operatorx<A>::mk_tx_(a_ptr, b % a);
                case __RRSHIFT__:
                    return operatorx<A>::mk_tx_(a_ptr, b >> a);
                }
            }
            throw std::runtime_error("Unknown/invalid elementwise binary operation");
        }
#pragma GCC diagnostic pop

    };
} // namespace x

tensor_i::future_type EWBinOp::op(EWBinOpId op, const tensor_i::future_type & a, const py::object & b)
{
    auto bb = x::mk_ftx(b);
    if(op == __MATMUL__) {
        return LinAlgOp::vecdot(a, bb, 0);
    }
    auto aa = std::move(a.get());
    auto bbb = std::move(bb.get());
    return  defer([op, aa, bbb](){
            return TypeDispatch<x::EWBinOp>(aa, bbb, op);
        });
}
