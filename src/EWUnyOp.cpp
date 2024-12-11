// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise unary ops.
*/

#include "sharpy/EWUnyOp.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/DepManager.hpp"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

namespace SHARPY {

// convert id of our unary op to id of imex::ndarray unary op
static mlir::Value createEWUnyOp(::mlir::OpBuilder &b,
                                 const ::mlir::Location &loc,
                                 const EWUnyOpId uop, mlir::ShapedType outTyp,
                                 mlir::Value a) {
  // this works only for static shapes
  switch (uop) {
  case __ABS__:
  case ABS:
    return b.create<mlir::tosa::AbsOp>(loc, outTyp, a);
  case ACOS:
    assert(false && "ACOS not implemented.");
  case ACOSH:
    assert(false && "ACOSH not implemented.");
  case ASIN:
    assert(false && "ASIN not implemented.");
  case ASINH:
    assert(false && "ASINH not implemented.");
  case ATAN:
    assert(false && "ATAN not implemented.");
  case ATANH:
    assert(false && "ATANH not implemented.");
  case CEIL:
    return b.create<mlir::tosa::CeilOp>(loc, outTyp, a);
  case COS:
    return b.create<mlir::tosa::CosOp>(loc, outTyp, a);
  case COSH:
    assert(false && "COSH not implemented.");
  case EXP:
    return b.create<mlir::tosa::ExpOp>(loc, outTyp, a);
  case EXPM1:
    assert(false && "EXPM1 not implemented.");
  case FLOOR:
    return b.create<mlir::tosa::FloorOp>(loc, outTyp, a);
  case ISFINITE:
    assert(false && "ISFINITE not implemented.");
  case ISINF:
    assert(false && "ISINF not implemented.");
  case ISNAN:
    assert(false && "ISNAN not implemented.");
  case LOG:
    return b.create<mlir::tosa::LogOp>(loc, outTyp, a);
  case LOG1P:
    assert(false && "LOG1P not implemented.");
  case LOG2:
    assert(false && "LOG2 not implemented.");
  case LOG10:
    assert(false && "LOG10 not implemented.");
  case ROUND: {
    mlir::Value empty = b.create<mlir::tensor::EmptyOp>(
        loc, outTyp.getShape(), outTyp.getElementType());
    return b.create<mlir::linalg::RoundOp>(loc, outTyp, a, empty).getResult(0);
  }
  case SIGN:
    assert(false && "SIGN not implemented.");
  case SIN:
    return b.create<mlir::tosa::SinOp>(loc, outTyp, a);
  case SINH:
    assert(false && "SINH not implemented.");
  case SQUARE: {
    mlir::Value empty = b.create<mlir::tensor::EmptyOp>(
        loc, outTyp.getShape(), outTyp.getElementType());
    return b.create<mlir::linalg::SquareOp>(loc, outTyp, a, empty).getResult(0);
  }
  case SQRT: {
    mlir::Value empty = b.create<mlir::tensor::EmptyOp>(
        loc, outTyp.getShape(), outTyp.getElementType());
    b.create<mlir::linalg::SqrtOp>(loc, outTyp, a, empty).getResult(0);
  }
  case TAN:
    assert(false && "TAN not implemented.");
  case TANH:
    return b.create<mlir::tosa::TanhOp>(loc, outTyp, a);
  case TRUNC:
    assert(false && "TRUNC not implemented.");
  case ERF:
    return b.create<mlir::tosa::ErfOp>(loc, outTyp, a);
  case LOGICAL_NOT:
    return b.create<mlir::tosa::LogicalNotOp>(loc, outTyp, a);
  case __NEG__:
  case NEGATIVE:
    return b.create<mlir::tosa::NegateOp>(loc, outTyp, a);
  case __POS__:
  case POSITIVE:
    assert(false && "POSITIVE not implemented.");
  default:
    throw std::invalid_argument("Unknown/invalid elementwise unary operation");
  }
  return {};
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

    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(av.getType());
    auto outTyp = ::mlir::cast<::mlir::RankedTensorType>(
        aTyp.cloneWith(shape(), aTyp.getElementType()));

    auto res = createEWUnyOp(builder, loc, _op, outTyp, av);
    // positive op will be eliminated so it is equivalent to a view
    auto view = (_op == POSITIVE || _op == __POS__);

    dm.addVal(
        this->guid(), res,
        [this, view](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
          auto t = mk_tnsr(this->guid(), _dtype, this->shape(), this->device(),
                           this->team(), allocated, aligned, offset, sizes,
                           strides, std::move(loffs));
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
