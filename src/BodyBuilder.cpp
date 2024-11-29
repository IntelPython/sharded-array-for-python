// SPDX-License-Identifier: BSD-3-Clause

/*
    helper functions for building the body of linalg::generic/reduce
*/

#include "sharpy/BodyBuilder.hpp"
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>

namespace SHARPY {

// any genericOp body needs to close with a yield
// we also add a cast op to "typ" if needed
template <typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc,
                  ::mlir::Type typ, T val) {
  auto res = val;
  if (typ != res.getType()) {
    res = builder.create<::mlir::UnrealizedConversionCastOp>(loc, typ, res)
              .getResult(0);
  }
  (void)builder.create<mlir::linalg::YieldOp>(loc, res);
}

/// Trivial binop builders have simple equivalents in Arith.
/// The Arith ops are accepted as template arguments, one for ints and one for
/// floats. Currently only integers and floats are supported.
/// Currently unsigned int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivialBinary(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhs = imex::createCast(loc, builder, args[0], typ);
    auto rhs = imex::createCast(loc, builder, args[1], typ);
    if (typ.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        yield(builder, loc, typ,
              builder.create<IOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (typ.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ,
              builder.create<FOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 && "Found float type but binary op not defined for floats");
    } else {
      assert(0 && "Only integers and floats supported for binary ops");
    }
  };
}

/// get a body builder for given binary operation and result type.
/// Accepts a result type to insert a cast after the operation if needed
/// FIXME: add missing ops
BodyType getBodyBuilder(EWBinOpId binOp, ::mlir::Type typ) {
  switch (binOp) {
  case ADD:
    return buildTrivialBinary<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  case ATAN2:
    return buildTrivialBinary<void, mlir::math::Atan2Op>(typ);
  case FLOOR_DIVIDE:
    return buildTrivialBinary<mlir::arith::FloorDivSIOp>(typ);
  case MAXIMUM:
    return buildTrivialBinary<mlir::arith::MaxSIOp, mlir::arith::MaximumFOp>(
        typ);
  case MINIMUM:
    return buildTrivialBinary<mlir::arith::MinSIOp, mlir::arith::MinimumFOp>(
        typ);
  case MODULO:
    return buildTrivialBinary<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case MULTIPLY:
    return buildTrivialBinary<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  case POWER:
    return buildTrivialBinary<mlir::math::IPowIOp, mlir::math::PowFOp>(typ);
  case SUBTRACT:
    return buildTrivialBinary<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  case DIVIDE:
    return buildTrivialBinary<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>(
        typ);
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

} // namespace SHARPY
