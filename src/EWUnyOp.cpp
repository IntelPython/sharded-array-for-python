// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise unary ops.
*/

#include "sharpy/EWUnyOp.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>

namespace SHARPY {

// convert id of our unary op to id of imex::ndarray unary op
static ::imex::ndarray::EWUnyOpId sharpy(const EWUnyOpId uop) {
  switch (uop) {
  case __ABS__:
  case ABS:
    return ::imex::ndarray::ABS;
  case ACOS:
    return ::imex::ndarray::ACOS;
  case ACOSH:
    return ::imex::ndarray::ACOSH;
  case ASIN:
    return ::imex::ndarray::ASIN;
  case ASINH:
    return ::imex::ndarray::ASINH;
  case ATAN:
    return ::imex::ndarray::ATAN;
  case ATANH:
    return ::imex::ndarray::ATANH;
  case CEIL:
    return ::imex::ndarray::CEIL;
  case COS:
    return ::imex::ndarray::COS;
  case COSH:
    return ::imex::ndarray::COSH;
  case EXP:
    return ::imex::ndarray::EXP;
  case EXPM1:
    return ::imex::ndarray::EXPM1;
  case FLOOR:
    return ::imex::ndarray::FLOOR;
  case ISFINITE:
    return ::imex::ndarray::ISFINITE;
  case ISINF:
    return ::imex::ndarray::ISINF;
  case ISNAN:
    return ::imex::ndarray::ISNAN;
  case LOG:
    return ::imex::ndarray::LOG;
  case LOG1P:
    return ::imex::ndarray::LOG1P;
  case LOG2:
    return ::imex::ndarray::LOG2;
  case LOG10:
    return ::imex::ndarray::LOG10;
  case ROUND:
    return ::imex::ndarray::ROUND;
  case SIGN:
    return ::imex::ndarray::SIGN;
  case SIN:
    return ::imex::ndarray::SIN;
  case SINH:
    return ::imex::ndarray::SINH;
  case SQUARE:
    return ::imex::ndarray::SQUARE;
  case SQRT:
    return ::imex::ndarray::SQRT;
  case TAN:
    return ::imex::ndarray::TAN;
  case TANH:
    return ::imex::ndarray::TANH;
  case TRUNC:
    return ::imex::ndarray::TRUNC;
  case ERF:
    return ::imex::ndarray::ERF;
  case LOGICAL_NOT:
    return ::imex::ndarray::LOGICAL_NOT;
  case __NEG__:
  case NEGATIVE:
    return ::imex::ndarray::NEGATIVE;
  case __POS__:
  case POSITIVE:
    return ::imex::ndarray::POSITIVE;
  default:
    throw std::invalid_argument("Unknown/invalid elementwise unary operation");
  }
}

struct DeferredEWUnyOp : public Deferred {
  id_type _a;
  EWUnyOpId _op;

  DeferredEWUnyOp() = default;
  DeferredEWUnyOp(EWUnyOpId op, const array_i::future_type &a)
      : Deferred(a.dtype(), a.shape(), a.device(), a.team()), _a(a.guid()),
        _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, Registry::get(_a));

    auto aTyp = ::mlir::cast<::imex::ndarray::NDArrayType>(av.getType());
    auto outTyp = aTyp.cloneWith(shape(), aTyp.getElementType());

    auto ndOpId = sharpy(_op);
    auto uop = builder.create<::imex::ndarray::EWUnyOp>(
        loc, outTyp, builder.getI32IntegerAttr(ndOpId), av);
    // positive op will be eliminated so it is equivalent to a view
    auto view = ndOpId == ::imex::ndarray::POSITIVE;

    dm.addVal(this->guid(), uop,
              [this, view](
                  uint64_t rank, void *l_allocated, void *l_aligned,
                  intptr_t l_offset, const intptr_t *l_sizes,
                  const intptr_t *l_strides, void *o_allocated, void *o_aligned,
                  intptr_t o_offset, const intptr_t *o_sizes,
                  const intptr_t *o_strides, void *r_allocated, void *r_aligned,
                  intptr_t r_offset, const intptr_t *r_sizes,
                  const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
                auto t = mk_tnsr(this->guid(), _dtype, this->shape(),
                                 this->device(), this->team(), l_allocated,
                                 l_aligned, l_offset, l_sizes, l_strides,
                                 o_allocated, o_aligned, o_offset, o_sizes,
                                 o_strides, r_allocated, r_aligned, r_offset,
                                 r_sizes, r_strides, std::move(loffs));
                if (view && Registry::has(_a)) {
                  t->set_base(Registry::get(_a).get());
                }
                this->set_value(std::move(t));
              });
    return false;
  }

  FactoryId factory() const override { return F_EWUNYOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_op)>(_op);
  }
};

FutureArray *EWUnyOp::op(EWUnyOpId op, const FutureArray &a) {
  return new FutureArray(defer<DeferredEWUnyOp>(op, a.get()));
}

FACTORY_INIT(DeferredEWUnyOp, F_EWUNYOP);
} // namespace SHARPY
