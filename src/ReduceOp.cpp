// SPDX-License-Identifier: BSD-3-Clause

// Implementation of reduction operations

#include "sharpy/ReduceOp.hpp"
#include "sharpy/BodyBuilder.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/jit/DepManager.hpp"
#include <mlir/Dialect/Linalg/IR/Linalg.h>

#include <algorithm>
#include <numeric>

namespace SHARPY {

// convert id of our reduction op to id of imex::ndarray reduction op
static mlir::Value createReduceOp(::mlir::OpBuilder &b,
                                  const ::mlir::Location &loc,
                                  const ReduceOpId rop,
                                  mlir::ShapedType outType, mlir::Value a,
                                  dim_vec_type axes) {
  std::sort(axes.begin(), axes.end());
  if (axes.empty()) {
    auto rank = mlir::cast<::mlir::RankedTensorType>(a.getType()).getRank();
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }
  auto elType = outType.getElementType();
  EWBinOpId bop = EWBINOP_LAST;
  mlir::TypedAttr iAttr;
  switch (rop) {
  case PROD:
    bop = MULTIPLY;
    iAttr = b.getOneAttr(elType);
    break;
  case SUM:
    bop = ADD;
    iAttr = b.getZeroAttr(elType);
    break;
  case MAX:
    bop = MAXIMUM;
    if (elType.isInteger()) {
      iAttr = b.getIntegerAttr(elType, mlir::APInt::getSignedMinValue(
                                           elType.getIntOrFloatBitWidth()));
    } else {
      iAttr = b.getFloatAttr(
          elType,
          mlir::APFloat::getLargest(
              mlir::cast<mlir::FloatType>(elType).getFloatSemantics(), true));
    }
    break;
  case MIN:
    bop = MINIMUM;
    if (elType.isInteger()) {
      iAttr = b.getIntegerAttr(elType, mlir::APInt::getSignedMaxValue(
                                           elType.getIntOrFloatBitWidth()));
    } else {
      iAttr = b.getFloatAttr(
          elType,
          mlir::APFloat::getLargest(
              mlir::cast<mlir::FloatType>(elType).getFloatSemantics(), false));
    }
    break;
  case MEAN:
  case STD:
  case VAR:
    throw std::invalid_argument("Reduction operation not implemented.");
  default:
    throw std::invalid_argument("Unknown reduction operation.");
  }

  mlir::Value iVal = b.create<mlir::arith::ConstantOp>(loc, iAttr);
  auto bodyBuilder = getBodyBuilder(bop, elType);
  auto empty = b.create<mlir::tensor::EmptyOp>(loc, outType.getShape(),
                                               outType.getElementType());
  auto filled = b.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{iVal},
                                               mlir::ValueRange{empty})
                    .getResult(0);
  return b.create<mlir::linalg::ReduceOp>(loc, a, filled, axes, bodyBuilder)
      .getResult(0);
}

struct DeferredReduceOp : public Deferred {
  id_type _a;
  dim_vec_type _dim;
  ReduceOpId _op;

  DeferredReduceOp() = default;
  DeferredReduceOp(ReduceOpId op, const array_i::future_type &a,
                   const dim_vec_type &dim)
      : Deferred(a.dtype(), {}, a.device(), a.team()), // FIXME rank
        _a(a.guid()), _dim(dim), _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // FIXME reduction over individual dimensions is not supported
    auto av = dm.getDependent(builder, Registry::get(_a));
    // return type 0d with same dtype as input
    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(av.getType());
    auto outTyp = ::mlir::cast<::mlir::RankedTensorType>(
        aTyp.cloneWith(shape(), aTyp.getElementType()));
    // reduction op
    auto res = createReduceOp(builder, loc, _op, outTyp, av, _dim);

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                this->set_value(mk_tnsr(this->guid(), _dtype, this->shape(),
                                        this->device(), this->team(), allocated,
                                        aligned, offset, sizes, strides,
                                        std::move(loffs)));
              });
    return false;
  }

  FactoryId factory() const override { return F_REDUCEOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template container<sizeof(dim_vec_type::value_type)>(_dim, 8);
    ser.template value<sizeof(_op)>(_op);
  }
};

FutureArray *ReduceOp::op(ReduceOpId op, const FutureArray &a,
                          const dim_vec_type &dim) {
  return new FutureArray(defer<DeferredReduceOp>(op, a.get(), dim));
}

FACTORY_INIT(DeferredReduceOp, F_REDUCEOP);
} // namespace SHARPY
