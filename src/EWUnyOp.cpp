#include "ddptensor/EWUnyOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/TypeDispatch.hpp"

#if 0
namespace x {

    class EWUnyOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(EWUnyOpId uop, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            const auto & ax = a_ptr->xarray();
            if(a_ptr->is_sliced()) {
                const auto & av = xt::strided_view(ax, a_ptr->lslice());
                return do_op(uop, av, a_ptr);
            }
            return do_op(uop, ax, a_ptr);
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename T1, typename T>
        static ptr_type do_op(EWUnyOpId uop, const T1 & a, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            switch(uop) {
            case __ABS__:
            case ABS:
                return operatorx<T>::mk_tx_(a_ptr, xt::abs(a));
            case ACOS:
                return operatorx<T>::mk_tx_(a_ptr, xt::acos(a));
            case ACOSH:
                return operatorx<T>::mk_tx_(a_ptr, xt::acosh(a));
            case ASIN:
                return operatorx<T>::mk_tx_(a_ptr, xt::asin(a));
            case ASINH:
                return operatorx<T>::mk_tx_(a_ptr, xt::asinh(a));
            case ATAN:
                return operatorx<T>::mk_tx_(a_ptr, xt::atan(a));
            case ATANH:
                return operatorx<T>::mk_tx_(a_ptr, xt::atanh(a));
            case CEIL:
                return operatorx<T>::mk_tx_(a_ptr, xt::ceil(a));
            case COS:
                return operatorx<T>::mk_tx_(a_ptr, xt::cos(a));
            case COSH:
                return operatorx<T>::mk_tx_(a_ptr, xt::cosh(a));
            case EXP:
                return operatorx<T>::mk_tx_(a_ptr, xt::exp(a));
            case EXPM1:
                return operatorx<T>::mk_tx_(a_ptr, xt::expm1(a));
            case FLOOR:
                return operatorx<T>::mk_tx_(a_ptr, xt::floor(a));
            case ISFINITE:
                return operatorx<T>::mk_tx_(a_ptr, xt::isfinite(a));
            case ISINF:
                return operatorx<T>::mk_tx_(a_ptr, xt::isinf(a));
            case ISNAN:
                return operatorx<T>::mk_tx_(a_ptr, xt::isnan(a));
            case LOG:
                return operatorx<T>::mk_tx_(a_ptr, xt::log(a));
            case LOG1P:
                return operatorx<T>::mk_tx_(a_ptr, xt::log1p(a));
            case LOG2:
                return operatorx<T>::mk_tx_(a_ptr, xt::log2(a));
            case LOG10:
                return operatorx<T>::mk_tx_(a_ptr, xt::log10(a));
            case ROUND:
                return operatorx<T>::mk_tx_(a_ptr, xt::round(a));
            case SIGN:
                return operatorx<T>::mk_tx_(a_ptr, xt::sign(a));
            case SIN:
                return operatorx<T>::mk_tx_(a_ptr, xt::sin(a));
            case SINH:
                return operatorx<T>::mk_tx_(a_ptr, xt::sinh(a));
            case SQUARE:
                return operatorx<T>::mk_tx_(a_ptr, xt::square(a));
            case SQRT:
                return operatorx<T>::mk_tx_(a_ptr, xt::sqrt(a));
            case TAN:
                return operatorx<T>::mk_tx_(a_ptr, xt::tan(a));
            case TANH:
                return operatorx<T>::mk_tx_(a_ptr, xt::tanh(a));
            case TRUNC:
                return operatorx<T>::mk_tx_(a_ptr, xt::trunc(a));
            case ERF:
                return operatorx<T>::mk_tx_(a_ptr, xt::erf(a));
            case __NEG__:
            case NEGATIVE:
            case __POS__:
            case POSITIVE:
            case LOGICAL_NOT:
                // FIXME
                throw std::runtime_error("Unary operation not implemented");
            }
            if constexpr (std::is_integral<T>::value) {
                switch(uop) {
                case __INVERT__:
                case BITWISE_INVERT:
                    throw std::runtime_error("Unary operation not implemented");
                }
            }
            throw std::runtime_error("Unknown/invalid elementwise unary operation");
        }
#pragma GCC diagnostic pop

    };
} //namespace x
#endif // if 0

struct DeferredEWUnyOp : public Deferred {
  id_type _a;
  EWUnyOpId _op;

  DeferredEWUnyOp() = default;
  DeferredEWUnyOp(EWUnyOpId op, const tensor_i::future_type &a)
      : _a(a.id()), _op(op) {}

  void run() {
    // const auto a = std::move(Registry::get(_a).get());
    // set_value(std::move(TypeDispatch<x::EWUnyOp>(a, _op)));
  }

  FactoryId factory() const { return F_EWUNYOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_op)>(_op);
  }
};

ddptensor *EWUnyOp::op(EWUnyOpId op, const ddptensor &a) {
  return new ddptensor(defer<DeferredEWUnyOp>(op, a.get()));
}

FACTORY_INIT(DeferredEWUnyOp, F_EWUNYOP);
