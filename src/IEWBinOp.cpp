// SPDX-License-Identifier: BSD-3-Clause

/*
  Inplace elementwise binary ops.
*/

#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/TypeDispatch.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

// convert id of our binop to id of imex::ptensor binop
static ::imex::ptensor::EWBinOpId ddpt2mlir(const IEWBinOpId bop) {
  switch (bop) {
  case __IADD__:
    return ::imex::ptensor::ADD;
  case __IAND__:
    return ::imex::ptensor::BITWISE_AND;
  case __IFLOORDIV__:
    return ::imex::ptensor::FLOOR_DIVIDE;
  case __ILSHIFT__:
    return ::imex::ptensor::BITWISE_LEFT_SHIFT;
  case __IMOD__:
    return ::imex::ptensor::MODULO;
  case __IMUL__:
    return ::imex::ptensor::MULTIPLY;
  case __IOR__:
    return ::imex::ptensor::BITWISE_OR;
  case __IPOW__:
    return ::imex::ptensor::POWER;
  case __IRSHIFT__:
    return ::imex::ptensor::BITWISE_RIGHT_SHIFT;
  case __ISUB__:
    return ::imex::ptensor::SUBTRACT;
  case __ITRUEDIV__:
    return ::imex::ptensor::TRUE_DIVIDE;
  case __IXOR__:
    return ::imex::ptensor::BITWISE_XOR;
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
  DeferredIEWBinOp(IEWBinOpId op, const tensor_i::future_type &a,
                   const tensor_i::future_type &b)
      : Deferred(a.dtype(), a.rank(), a.balanced()), _a(a.id()), _b(b.id()),
        _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // FIXME the type of the result is based on a only
    auto av = dm.getDependent(builder, _a);
    auto bv = dm.getDependent(builder, _b);

    auto aTyp = ::imex::dist::getPTensorType(av);
    ::mlir::SmallVector<int64_t> shape(rank(), ::mlir::ShapedType::kDynamic);
    auto outTyp =
        ::imex::ptensor::PTensorType::get(shape, aTyp.getElementType());

    auto binop = builder.create<::imex::ptensor::EWBinOp>(
        loc, outTyp, builder.getI32IntegerAttr(ddpt2mlir(_op)), av, bv);
    // insertsliceop has no return value, so we just create the op...
    auto zero = ::imex::createIndex(loc, builder, 0);
    auto one = ::imex::createIndex(loc, builder, 1);
    auto dyn = ::imex::createIndex(loc, builder, ::mlir::ShapedType::kDynamic);
    ::mlir::SmallVector<::mlir::Value> offs(rank(), zero);
    ::mlir::SmallVector<::mlir::Value> szs(rank(), dyn);
    ::mlir::SmallVector<::mlir::Value> strds(rank(), one);
    (void)builder.create<::imex::ptensor::InsertSliceOp>(loc, av, binop, offs,
                                                         szs, strds);
    // ... and use av as to later create the ptensor
    dm.addVal(this->guid(), av,
              [this](Transceiver *transceiver, uint64_t rank, void *allocated,
                     void *aligned, intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, uint64_t *gs_allocated,
                     uint64_t *gs_aligned, uint64_t *lo_allocated,
                     uint64_t *lo_aligned, uint64_t balanced) {
                this->set_value(Registry::get(this->_a).get());
              });
    return false;
  }

  FactoryId factory() const { return F_IEWBINOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_b)>(_b);
    ser.template value<sizeof(_op)>(_op);
  }
};

ddptensor *IEWBinOp::op(IEWBinOpId op, ddptensor &a, const py::object &b) {
  auto bb = Creator::mk_future(b);
  auto res =
      new ddptensor(defer<DeferredIEWBinOp>(op, a.get(), bb.first->get()));
  if (bb.second)
    delete bb.first;
  return res;
}

FACTORY_INIT(DeferredIEWBinOp, F_IEWBINOP);
