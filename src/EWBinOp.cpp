// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise binary ops.
*/

#include "sharpy/EWBinOp.hpp"
#include "sharpy/BodyBuilder.hpp"
#include "sharpy/Broadcast.hpp"
#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/LinAlgOp.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Registry.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

static bool is_reflected_op(EWBinOpId op) {
  switch (op) {
  case __RADD__:
  case __RAND__:
  case __RFLOORDIV__:
  case __RLSHIFT__:
  case __RMOD__:
  case __RMUL__:
  case __ROR__:
  case __RPOW__:
  case __RRSHIFT__:
  case __RSUB__:
  case __RTRUEDIV__:
  case __RXOR__:
    return true;
  default:
    return false;
  }
}

static bool is_inplace_op(EWBinOpId op) {
  switch (op) {
  case __IADD__:
  case __IAND__:
  case __IFLOORDIV__:
  case __ILSHIFT__:
  case __IMOD__:
  case __IMUL__:
  case __IOR__:
  case __IPOW__:
  case __IRSHIFT__:
  case __ISUB__:
  case __ITRUEDIV__:
  case __IXOR__:
    return true;
  default:
    return false;
  }
}

// create a linalg.generic for the given binary operation
static mlir::Value createLinalgGeneric(::mlir::OpBuilder &b,
                                       const ::mlir::Location &loc,
                                       const EWBinOpId bop,
                                       mlir::ShapedType outType,
                                       mlir::Value lhs, mlir::Value rhs) {
  // create output tensor with right dimensions
  auto tensor = b.create<mlir::tensor::EmptyOp>(loc, outType.getShape(),
                                                outType.getElementType())
                    .getResult();
  auto lhsType = mlir::cast<mlir::ShapedType>(lhs.getType());
  auto rhsType = mlir::cast<mlir::ShapedType>(rhs.getType());

  // we need affine maps for linalg::generic
  // as long as we have no proper support for rank-reduced sizes above
  // Linalg, we can handle only
  //   - explicitly rank-reduced inputs (such as explicit 0d tensors)
  //   - shapes with static dim-sizes of 1
  ::mlir::SmallVector<::mlir::AffineExpr> lhsExprs, rhsExprs, resExprs;
  for (int i = 0; i < lhsType.getRank(); ++i) {
    lhsExprs.emplace_back(lhsType.getShape()[i] == 1
                              ? b.getAffineConstantExpr(0)
                              : b.getAffineDimExpr(i));
  }
  for (int i = 0; i < rhsType.getRank(); ++i) {
    rhsExprs.emplace_back(rhsType.getShape()[i] == 1
                              ? b.getAffineConstantExpr(0)
                              : b.getAffineDimExpr(i));
  }
  for (unsigned i = 0; i < outType.getRank(); ++i) {
    resExprs.emplace_back(b.getAffineDimExpr(i));
  }
  auto lhsMap = ::mlir::AffineMap::get(outType.getRank(), /*symbolCount=*/0,
                                       lhsExprs, b.getContext());
  auto rhsMap = ::mlir::AffineMap::get(outType.getRank(), /*symbolCount=*/0,
                                       rhsExprs, b.getContext());
  auto resMap = b.getMultiDimIdentityMap(outType.getRank());

  // we just make all dims parallel
  ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
      outType.getRank(), ::mlir::utils::IteratorType::parallel);

  return b
      .create<::mlir::linalg::GenericOp>(
          loc, tensor.getType(), ::mlir::ValueRange{lhs, rhs}, tensor,
          ::mlir::ArrayRef<::mlir::AffineMap>{lhsMap, rhsMap, resMap},
          iterators, getBodyBuilder(bop, outType.getElementType()))
      .getResult(0);
}

// convert id of our binop to id of imex::ndarray binop
static mlir::Value createEWBinOp(::mlir::OpBuilder &b,
                                 const ::mlir::Location &loc,
                                 const EWBinOpId bop, mlir::ShapedType outTyp,
                                 mlir::Value lhs, mlir::Value rhs) {
  // this works only for static shapes
  switch (bop) {
  // cases handled by tosa
  case __ADD__:
  case ADD:
  case __RADD__:
  case __IADD__:
    return b.create<mlir::tosa::AddOp>(loc, outTyp, lhs, rhs);
  case __SUB__:
  case SUBTRACT:
  case __RSUB__:
  case __ISUB__:
    return b.create<mlir::tosa::SubOp>(loc, outTyp, lhs, rhs);
  case __POW__:
  case POWER:
  case POW:
  case __RPOW__:
  case __IPOW__:
    return b.create<mlir::tosa::PowOp>(loc, outTyp, lhs, rhs);
  case __LSHIFT__:
  case BITWISE_LEFT_SHIFT:
  case __RLSHIFT__:
  case __ILSHIFT__:
    return b.create<mlir::tosa::LogicalLeftShiftOp>(loc, outTyp, lhs, rhs);
  case __RSHIFT__:
  case BITWISE_RIGHT_SHIFT:
  case __RRSHIFT__:
  case __IRSHIFT__:
    return b.create<mlir::tosa::LogicalRightShiftOp>(loc, outTyp, lhs, rhs);
  case __AND__:
  case BITWISE_AND:
  case __RAND__:
  case __IAND__:
    return b.create<mlir::tosa::BitwiseAndOp>(loc, outTyp, lhs, rhs);
  case __OR__:
  case BITWISE_OR:
  case __ROR__:
  case __IOR__:
    return b.create<mlir::tosa::BitwiseOrOp>(loc, outTyp, lhs, rhs);
  case __XOR__:
  case BITWISE_XOR:
  case __RXOR__:
  case __IXOR__:
    return b.create<mlir::tosa::BitwiseXorOp>(loc, outTyp, lhs, rhs);
  // cases handled by linalg
  case __MUL__:
  case MULTIPLY:
  case __RMUL__:
  case __IMUL__: {
    mlir::Value empty = b.create<mlir::tensor::EmptyOp>(
        loc, outTyp.getShape(), outTyp.getElementType());
    return b
        .create<mlir::linalg::MulOp>(loc, outTyp, mlir::ValueRange{lhs, rhs},
                                     empty)
        .getResult(0);
  }
  case __TRUEDIV__:
  case DIVIDE:
  case __RTRUEDIV__:
  case __ITRUEDIV__: {
    mlir::Value empty = b.create<mlir::tensor::EmptyOp>(
        loc, outTyp.getShape(), outTyp.getElementType());
    return b
        .create<mlir::linalg::DivOp>(loc, outTyp, mlir::ValueRange{lhs, rhs},
                                     empty)
        .getResult(0);
  }
  case ATAN2:
    createLinalgGeneric(b, loc, bop, outTyp, lhs, rhs);
  case __FLOORDIV__:
  case FLOOR_DIVIDE:
  case __RFLOORDIV__:
  case __IFLOORDIV__:
    createLinalgGeneric(b, loc, FLOOR_DIVIDE, outTyp, lhs, rhs);
  case LOGADDEXP:
    createLinalgGeneric(b, loc, bop, outTyp, lhs, rhs);
  case __MOD__:
  case REMAINDER:
  case MODULO:
  case __RMOD__:
  case __IMOD__:
    createLinalgGeneric(b, loc, MODULO, outTyp, lhs, rhs);
  default:
    throw std::invalid_argument("Unknown/invalid elementwise binary operation");
  }
  // __MATMUL__ is handled before dispatching, see below
  return {};
}

struct DeferredEWBinOp : public Deferred {
  id_type _a;
  id_type _b;
  EWBinOpId _op;

  DeferredEWBinOp() = default;
  DeferredEWBinOp(EWBinOpId op, const array_i::future_type &a,
                  const array_i::future_type &b)
      : Deferred(promoted_dtype(a.dtype(), b.dtype()),
                 broadcast(a.shape(), b.shape()), a.device(), a.team()),
        _a(a.guid()), _b(b.guid()), _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, Registry::get(_a));
    auto bv = dm.getDependent(builder, Registry::get(_b));

    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(av.getType());
    auto outElemType = jit::getMLIRType(builder, _dtype);
    auto outTyp = ::mlir::cast<::mlir::RankedTensorType>(
        aTyp.cloneWith(shape(), outElemType));

    auto isInplace = is_inplace_op(_op);
    mlir::Value res;
    if (isInplace || !is_reflected_op(_op)) {
      res = createEWBinOp(builder, loc, _op, outTyp, av, bv);
      if (isInplace) {
        // insertsliceop has no return value, so we just create the op...
        auto zero = ::imex::createIndex(loc, builder, 0);
        auto one = ::imex::createIndex(loc, builder, 1);
        auto dyn =
            ::imex::createIndex(loc, builder, ::mlir::ShapedType::kDynamic);
        ::mlir::SmallVector<::mlir::Value> offs(rank(), zero);
        ::mlir::SmallVector<::mlir::Value> szs(rank(), dyn);
        ::mlir::SmallVector<::mlir::Value> strds(rank(), one);
        (void)builder.create<::imex::ndarray::InsertSliceOp>(loc, av, res, offs,
                                                             szs, strds);
        res = av;
      }
    } else {
      res = createEWBinOp(builder, loc, _op, outTyp, bv, av);
    }

    dm.addVal(this->guid(), res,
              [this, isInplace](uint64_t rank, void *allocated, void *aligned,
                                intptr_t offset, const intptr_t *sizes,
                                const intptr_t *strides,
                                std::vector<int64_t> &&loffs) {
                if (isInplace) {
                  this->set_value(Registry::get(this->_a).get());
                } else {
                  this->set_value(mk_tnsr(this->guid(), _dtype, this->shape(),
                                          this->device(), this->team(),
                                          allocated, aligned, offset, sizes,
                                          strides, std::move(loffs)));
                }
              });
    return false;
  }

  FactoryId factory() const override { return F_EWBINOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_b)>(_b);
    ser.template value<sizeof(_op)>(_op);
  }
};

FutureArray *EWBinOp::op(EWBinOpId op, py::object &a, const py::object &b) {
  std::string deva, devb;
  std::string teama = 0, teamb = 0;
  DTypeId dtypea = DTYPE_LAST, dtypeb = DTYPE_LAST;

  if (py::isinstance<FutureArray>(a)) {
    auto tmp = a.cast<FutureArray *>()->get();
    deva = tmp.device();
    teama = tmp.team();
    dtypea = tmp.dtype();
  }
  if (py::isinstance<FutureArray>(b)) {
    auto tmp = b.cast<FutureArray *>()->get();
    devb = tmp.device();
    teamb = tmp.team();
    dtypeb = tmp.dtype();
  }
  auto aa = Creator::mk_future(a, devb, teamb, dtypeb);
  auto bb = Creator::mk_future(b, deva, teama, dtypea);
  if (bb.first->get().device() != aa.first->get().device()) {
    throw std::runtime_error(
        "devices of operands do not match in binary operation");
  }
  if (bb.first->get().team() != aa.first->get().team()) {
    throw std::runtime_error(
        "teams of operands do not match in binary operation");
  }
  if (op == __MATMUL__) {
    return LinAlgOp::vecdot(*aa.first, *bb.first, 0);
  }
  auto res = new FutureArray(
      defer<DeferredEWBinOp>(op, aa.first->get(), bb.first->get()));
  if (aa.second)
    delete aa.first;
  if (bb.second)
    delete bb.first;
  return res;
}

FACTORY_INIT(DeferredEWBinOp, F_EWBINOP);
} // namespace SHARPY
