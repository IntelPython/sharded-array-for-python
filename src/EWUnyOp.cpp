// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise unary ops.
*/

#include "ddptensor/EWUnyOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/TypeDispatch.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>

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

// convert id of our unary op to id of imex::ptensor unary op
static ::imex::ptensor::EWUnyOpId ddpt2mlir(const EWUnyOpId uop) {
  switch (uop) {
  case __ABS__:
  case ABS:
    return ::imex::ptensor::ABS;
  case ACOS:
    return ::imex::ptensor::ACOS;
  case ACOSH:
    return ::imex::ptensor::ACOSH;
  case ASIN:
    return ::imex::ptensor::ASIN;
  case ASINH:
    return ::imex::ptensor::ASINH;
  case ATAN:
    return ::imex::ptensor::ATAN;
  case ATANH:
    return ::imex::ptensor::ATANH;
  case CEIL:
    return ::imex::ptensor::CEIL;
  case COS:
    return ::imex::ptensor::COS;
  case COSH:
    return ::imex::ptensor::COSH;
  case EXP:
    return ::imex::ptensor::EXP;
  case EXPM1:
    return ::imex::ptensor::EXPM1;
  case FLOOR:
    return ::imex::ptensor::FLOOR;
  case ISFINITE:
    return ::imex::ptensor::ISFINITE;
  case ISINF:
    return ::imex::ptensor::ISINF;
  case ISNAN:
    return ::imex::ptensor::ISNAN;
  case LOG:
    return ::imex::ptensor::LOG;
  case LOG1P:
    return ::imex::ptensor::LOG1P;
  case LOG2:
    return ::imex::ptensor::LOG2;
  case LOG10:
    return ::imex::ptensor::LOG10;
  case ROUND:
    return ::imex::ptensor::ROUND;
  case SIGN:
    return ::imex::ptensor::SIGN;
  case SIN:
    return ::imex::ptensor::SIN;
  case SINH:
    return ::imex::ptensor::SINH;
  case SQUARE:
    return ::imex::ptensor::SQUARE;
  case SQRT:
    return ::imex::ptensor::SQRT;
  case TAN:
    return ::imex::ptensor::TAN;
  case TANH:
    return ::imex::ptensor::TANH;
  case TRUNC:
    return ::imex::ptensor::TRUNC;
  case ERF:
    return ::imex::ptensor::ERF;
  case LOGICAL_NOT:
    return ::imex::ptensor::LOGICAL_NOT;
  case __NEG__:
  case NEGATIVE:
  case __POS__:
  case POSITIVE:
  default:
    throw std::runtime_error("Unknown/invalid elementwise unary operation");
  }
}

struct DeferredEWUnyOp : public Deferred {
  id_type _a;
  EWUnyOpId _op;

  DeferredEWUnyOp() = default;
  DeferredEWUnyOp(EWUnyOpId op, const tensor_i::future_type &a)
      : Deferred(a.dtype(), a.shape(), a.team(), true), _a(a.guid()), _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, _a);

    auto outTyp = ::imex::ptensor::PTensorType::get(
        shape(), ::imex::dist::getElementType(av));

    auto uop = builder.create<::imex::ptensor::EWUnyOp>(
        loc, outTyp, builder.getI32IntegerAttr(ddpt2mlir(_op)), av);
    dm.addVal(this->guid(), uop,
              [this](Transceiver *transceiver, uint64_t rank, void *l_allocated,
                     void *l_aligned, intptr_t l_offset,
                     const intptr_t *l_sizes, const intptr_t *l_strides,
                     void *o_allocated, void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                this->set_value(std::move(mk_tnsr(
                    transceiver, _dtype, this->shape(), l_allocated, l_aligned,
                    l_offset, l_sizes, l_strides, o_allocated, o_aligned,
                    o_offset, o_sizes, o_strides, r_allocated, r_aligned,
                    r_offset, r_sizes, r_strides, lo_allocated, lo_aligned)));
              });
    return false;
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
