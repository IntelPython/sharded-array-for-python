// SPDX-License-Identifier: BSD-3-Clause

/*
  Inplace elementwise binary ops.
*/

#include "sharpy/IEWBinOp.hpp"
#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Registry.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

namespace SHARPY {

// convert id of our binop to id of imex::ndarray binop
static ::imex::ndarray::EWBinOpId sharpy2mlir(const IEWBinOpId bop) {
  switch (bop) {
  case __IADD__:
    return ::imex::ndarray::ADD;
  case __IAND__:
    return ::imex::ndarray::BITWISE_AND;
  case __IFLOORDIV__:
    return ::imex::ndarray::FLOOR_DIVIDE;
  case __ILSHIFT__:
    return ::imex::ndarray::BITWISE_LEFT_SHIFT;
  case __IMOD__:
    return ::imex::ndarray::MODULO;
  case __IMUL__:
    return ::imex::ndarray::MULTIPLY;
  case __IOR__:
    return ::imex::ndarray::BITWISE_OR;
  case __IPOW__:
    return ::imex::ndarray::POWER;
  case __IRSHIFT__:
    return ::imex::ndarray::BITWISE_RIGHT_SHIFT;
  case __ISUB__:
    return ::imex::ndarray::SUBTRACT;
  case __ITRUEDIV__:
    return ::imex::ndarray::TRUE_DIVIDE;
  case __IXOR__:
    return ::imex::ndarray::BITWISE_XOR;
  default:
    throw std::runtime_error(
        "Unknown/invalid inplace elementwise binary operation");
  }
}

struct DeferredIEWBinOp : public Deferred {
  id_type _a;
  id_type _b;
  IEWBinOpId _op;

  DeferredIEWBinOp() = default;
  DeferredIEWBinOp(IEWBinOpId op, const array_i::future_type &a,
                   const array_i::future_type &b)
      : Deferred(a.dtype(), a.shape(), a.device(), a.team()), _a(a.guid()),
        _b(b.guid()), _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // FIXME the type of the result is based on a only
    auto av = dm.getDependent(builder, Registry::get(_a));
    auto bv = dm.getDependent(builder, Registry::get(_b));

    auto aTyp = av.getType().cast<::imex::ndarray::NDArrayType>();
    auto outTyp = aTyp.cloneWith(shape(), aTyp.getElementType());

    auto binop = builder.create<::imex::ndarray::EWBinOp>(
        loc, outTyp, builder.getI32IntegerAttr(sharpy2mlir(_op)), av, bv);
    // insertsliceop has no return value, so we just create the op...
    auto zero = ::imex::createIndex(loc, builder, 0);
    auto one = ::imex::createIndex(loc, builder, 1);
    auto dyn = ::imex::createIndex(loc, builder, ::mlir::ShapedType::kDynamic);
    ::mlir::SmallVector<::mlir::Value> offs(rank(), zero);
    ::mlir::SmallVector<::mlir::Value> szs(rank(), dyn);
    ::mlir::SmallVector<::mlir::Value> strds(rank(), one);
    (void)builder.create<::imex::ndarray::InsertSliceOp>(loc, av, binop, offs,
                                                         szs, strds);
    // ... and use av as to later create the ndarray
    dm.addVal(this->guid(), av,
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     std::vector<int64_t> &&loffs) {
                this->set_value(Registry::get(this->_a).get());
              });
    return false;
  }

  FactoryId factory() const override { return F_IEWBINOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_b)>(_b);
    ser.template value<sizeof(_op)>(_op);
  }
};

FutureArray *IEWBinOp::op(IEWBinOpId op, FutureArray &a, const py::object &b) {
  auto bb =
      Creator::mk_future(b, a.get().device(), a.get().team(), a.get().dtype());
  auto res =
      new FutureArray(defer<DeferredIEWBinOp>(op, a.get(), bb.first->get()));
  if (bb.second)
    delete bb.first;
  return res;
}

FACTORY_INIT(DeferredIEWBinOp, F_IEWBINOP);
} // namespace SHARPY
