#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

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
            const auto _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            const auto & a = xt::strided_view(_a->xarray(), _a->lslice());
            
            switch(uop) {
            case __ABS__:
            case ABS:
                return operatorx<T>::mk_tx_(*_a, xt::abs(a));
            case ACOS:
                return operatorx<T>::mk_tx_(*_a, xt::acos(a));
            case ACOSH:
                return operatorx<T>::mk_tx_(*_a, xt::acosh(a));
            case ASIN:
                return operatorx<T>::mk_tx_(*_a, xt::asin(a));
            case ASINH:
                return operatorx<T>::mk_tx_(*_a, xt::asinh(a));
            case ATAN:
                return operatorx<T>::mk_tx_(*_a, xt::atan(a));
            case ATANH:
                return operatorx<T>::mk_tx_(*_a, xt::atanh(a));
            case CEIL:
                return operatorx<T>::mk_tx_(*_a, xt::ceil(a));
            case COS:
                return operatorx<T>::mk_tx_(*_a, xt::cos(a));
            case COSH:
                return operatorx<T>::mk_tx_(*_a, xt::cosh(a));
            case EXP:
                return operatorx<T>::mk_tx_(*_a, xt::exp(a));
            case EXPM1:
                return operatorx<T>::mk_tx_(*_a, xt::expm1(a));
            case FLOOR:
                return operatorx<T>::mk_tx_(*_a, xt::floor(a));
            case ISFINITE:
                return operatorx<T>::mk_tx_(*_a, xt::isfinite(a));
            case ISINF:
                return operatorx<T>::mk_tx_(*_a, xt::isinf(a));
            case ISNAN:
                return operatorx<T>::mk_tx_(*_a, xt::isnan(a));
            case LOG:
                return operatorx<T>::mk_tx_(*_a, xt::log(a));
            case LOG1P:
                return operatorx<T>::mk_tx_(*_a, xt::log1p(a));
            case LOG2:
                return operatorx<T>::mk_tx_(*_a, xt::log2(a));
            case LOG10:
                return operatorx<T>::mk_tx_(*_a, xt::log10(a));
            case ROUND:
                return operatorx<T>::mk_tx_(*_a, xt::round(a));
            case SIGN:
                return operatorx<T>::mk_tx_(*_a, xt::sign(a));
            case SIN:
                return operatorx<T>::mk_tx_(*_a, xt::sin(a));
            case SINH:
                return operatorx<T>::mk_tx_(*_a, xt::sinh(a));
            case SQUARE:
                return operatorx<T>::mk_tx_(*_a, xt::square(a));
            case SQRT:
                return operatorx<T>::mk_tx_(*_a, xt::sqrt(a));
            case TAN:
                return operatorx<T>::mk_tx_(*_a, xt::tan(a));
            case TANH:
                return operatorx<T>::mk_tx_(*_a, xt::tanh(a));
            case TRUNC:
                return operatorx<T>::mk_tx_(*_a, xt::trunc(a));
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
} //namespace x

tensor_i::ptr_type EWUnyOp::op(EWUnyOpId op, x::DPTensorBaseX::ptr_type a)
{
    return TypeDispatch<x::EWUnyOp>(a->dtype(), op, a);
}
