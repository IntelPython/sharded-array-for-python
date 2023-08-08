// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise binary ops.
*/

#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/Broadcast.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/TypePromotion.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>

// convert id of our binop to id of imex::ptensor binop
static ::imex::ptensor::EWBinOpId ddpt2mlir(const EWBinOpId bop) {
  switch (bop) {
  case __ADD__:
  case ADD:
  case __RADD__:
    return ::imex::ptensor::ADD;
  case ATAN2:
    return ::imex::ptensor::ATAN2;
  case __FLOORDIV__:
  case FLOOR_DIVIDE:
  case __RFLOORDIV__:
    return ::imex::ptensor::FLOOR_DIVIDE;
    // __MATMUL__ is handled before dispatching, see below
  case __MUL__:
  case MULTIPLY:
  case __RMUL__:
    return ::imex::ptensor::MULTIPLY;
  case __SUB__:
  case SUBTRACT:
  case __RSUB__:
    return ::imex::ptensor::SUBTRACT;
  case __TRUEDIV__:
  case DIVIDE:
  case __RTRUEDIV__:
    return ::imex::ptensor::TRUE_DIVIDE;
  case __POW__:
  case POW:
  case __RPOW__:
    return ::imex::ptensor::POWER;
  case LOGADDEXP:
    return ::imex::ptensor::LOGADDEXP;
  case __LSHIFT__:
  case BITWISE_LEFT_SHIFT:
  case __RLSHIFT__:
    return ::imex::ptensor::BITWISE_LEFT_SHIFT;
  case __MOD__:
  case REMAINDER:
  case __RMOD__:
    return ::imex::ptensor::MODULO;
  case __RSHIFT__:
  case BITWISE_RIGHT_SHIFT:
  case __RRSHIFT__:
    return ::imex::ptensor::BITWISE_RIGHT_SHIFT;
  case __AND__:
  case BITWISE_AND:
  case __RAND__:
    return ::imex::ptensor::BITWISE_AND;
  case __OR__:
  case BITWISE_OR:
  case __ROR__:
    return ::imex::ptensor::BITWISE_OR;
  case __XOR__:
  case BITWISE_XOR:
  case __RXOR__:
    return ::imex::ptensor::BITWISE_XOR;
  default:
    throw std::runtime_error("Unknown/invalid elementwise binary operation");
  }
}

struct DeferredEWBinOp : public Deferred {
  id_type _a;
  id_type _b;
  EWBinOpId _op;

  DeferredEWBinOp() = default;
  DeferredEWBinOp(EWBinOpId op, const tensor_i::future_type &a,
                  const tensor_i::future_type &b)
      : Deferred(a.dtype(), broadcast(a.shape(), b.shape()), a.team(), true),
        _a(a.guid()), _b(b.guid()), _op(op) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // FIXME the type of the result is based on a only
    auto av = dm.getDependent(builder, _a);
    auto bv = dm.getDependent(builder, _b);

    auto outTyp = ::imex::ptensor::PTensorType::get(
        shape(), ::imex::dist::getElementType(av));

    auto bop = builder.create<::imex::ptensor::EWBinOp>(
        loc, outTyp, builder.getI32IntegerAttr(ddpt2mlir(_op)), av, bv);
    // auto bop =
    //     builder.create<::imex::ptensor::EWBinOp>(loc, ddpt2mlir(_op), av,
    //     bv);
    dm.addVal(this->guid(), bop,
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

  FactoryId factory() const { return F_EWBINOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_b)>(_b);
    ser.template value<sizeof(_op)>(_op);
  }
};

ddptensor *EWBinOp::op(EWBinOpId op, const py::object &a, const py::object &b) {
  uint64_t teama = 0, teamb = 0;
  if (py::isinstance<ddptensor>(a))
    teama = a.cast<ddptensor *>()->get().team();
  else if (py::isinstance<ddptensor>(b))
    teamb = b.cast<ddptensor *>()->get().team();
  auto team = teama ? teama : teamb;
  auto bb = Creator::mk_future(b, team);
  auto aa = Creator::mk_future(a, team);
  if (bb.first->get().team() != aa.first->get().team()) {
    throw std::runtime_error(
        "teams of operands do not match in binary operation");
  }
  if (op == __MATMUL__) {
    return LinAlgOp::vecdot(*aa.first, *bb.first, 0);
  }
  auto res = new ddptensor(
      defer<DeferredEWBinOp>(op, aa.first->get(), bb.first->get()));
  if (aa.second)
    delete aa.first;
  if (bb.second)
    delete bb.first;
  return res;
}

FACTORY_INIT(DeferredEWBinOp, F_EWBINOP);
