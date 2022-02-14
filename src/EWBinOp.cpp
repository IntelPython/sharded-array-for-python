#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

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
        static ptr_type integral_op(EWBinOpId iop, const DPTensorX<T> & tx, A && a, B && b)
        {
            switch(iop) {
            case __AND__:
            case BITWISE_AND:
                return operatorx<T>::mk_tx_(tx, a & b);
            case __RAND__:
                return operatorx<T>::mk_tx_(tx, b & a);
            case __LSHIFT__:
            case BITWISE_LEFT_SHIFT:
                return operatorx<T>::mk_tx_(tx, a << b);
            case __MOD__:
            case REMAINDER:
                return operatorx<T>::mk_tx_(tx, a % b);
            case __OR__:
            case BITWISE_OR:
                return operatorx<T>::mk_tx_(tx, a | b);
            case __ROR__:
                return operatorx<T>::mk_tx_(tx, b | a);
            case __RSHIFT__:
            case BITWISE_RIGHT_SHIFT:
                return operatorx<T>::mk_tx_(tx, a >> b);
            case __XOR__:
            case BITWISE_XOR:
                return operatorx<T>::mk_tx_(tx, a ^ b);
            case __RXOR__:
                return operatorx<T>::mk_tx_(tx, b ^ a);
            case __RLSHIFT__:
                return operatorx<T>::mk_tx_(tx, b << a);
            case __RMOD__:
                return operatorx<T>::mk_tx_(tx, b % a);
            case __RRSHIFT__:
                return operatorx<T>::mk_tx_(tx, b >> a);
            default:
                throw std::runtime_error("Unknown elementwise binary operation");
            }
        }

        static ptr_type op(EWBinOpId bop, const ptr_type & a_ptr, const ptr_type & b_ptr)
        {
            const auto _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            const auto _b = dynamic_cast<DPTensorX<T>*>(b_ptr.get());
            if(!_a || !_b)
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            const auto & a = xt::strided_view(_a->xarray(), _a->lslice());
            const auto & b = xt::strided_view(_b->xarray(), _b->lslice());
            
            switch(bop) {
            case __ADD__:
            case ADD:
                return operatorx<T>::mk_tx_(*_a, a + b);
            case __RADD__:
                return operatorx<T>::mk_tx_(*_a, b + a);
            case ATAN2:
                return  operatorx<T>::mk_tx_(*_a, xt::atan2(a, b));
            case __EQ__:
            case EQUAL:
                return  operatorx<T>::mk_tx_(*_a, xt::equal(a, b));
            case __FLOORDIV__:
            case FLOOR_DIVIDE:
                return operatorx<T>::mk_tx_(*_a, xt::floor(a / b));
            case __GE__:
            case GREATER_EQUAL:
                return operatorx<T>::mk_tx_(*_a, a >= b);
            case __GT__:
            case GREATER:
                return operatorx<T>::mk_tx_(*_a, a > b);
            case __LE__:
            case LESS_EQUAL:
                return operatorx<T>::mk_tx_(*_a, a <= b);
            case __LT__:
            case LESS:
                return operatorx<T>::mk_tx_(*_a, a < b);
            case __MUL__:
            case MULTIPLY:
                return operatorx<T>::mk_tx_(*_a, a * b);
            case __RMUL__:
                return operatorx<T>::mk_tx_(*_a, b * a);
            case __NE__:
            case NOT_EQUAL:
                return operatorx<T>::mk_tx_(*_a, xt::not_equal(a, b));
            case __SUB__:
            case SUBTRACT:
                return operatorx<T>::mk_tx_(*_a, a - b);
            case __TRUEDIV__:
            case DIVIDE:
                return operatorx<T>::mk_tx_(*_a, a / b);
            case __RFLOORDIV__:
                return operatorx<T>::mk_tx_(*_a, xt::floor(b / a));
            case __RSUB__:
                return operatorx<T>::mk_tx_(*_a, b - a);
            case __RTRUEDIV__:
                return operatorx<T>::mk_tx_(*_a, b / a);
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
} // namespace x
    
tensor_i::ptr_type EWBinOp::op(EWBinOpId op, x::DPTensorBaseX::ptr_type a, x::DPTensorBaseX::ptr_type b)
{
    return TypeDispatch<x::EWBinOp>(a->dtype(), op, a, b);
}
