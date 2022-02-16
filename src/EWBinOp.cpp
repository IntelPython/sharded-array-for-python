#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

    class EWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename A, typename B>
        static ptr_type op(EWBinOpId bop, const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            const auto & a = xt::strided_view(a_ptr->xarray(), a_ptr->lslice());
            const auto & b = xt::strided_view(b_ptr->xarray(), b_ptr->lslice());
            
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
            if constexpr (std::is_integral<A>::value && std::is_integral<B>::value) {
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
    
tensor_i::ptr_type EWBinOp::op(EWBinOpId op, x::DPTensorBaseX::ptr_type a, py::object b)
{
    auto bb = x::mk_tx(b);
    return TypeDispatch<x::EWBinOp>(a, bb, op);
}
