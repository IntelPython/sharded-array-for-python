// SPDX-License-Identifier: BSD-3-Clause

/*
  Elementwise binary ops.
*/

#include "sharpy/EWBinOp.hpp"
#include "sharpy/Broadcast.hpp"
#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/LinAlgOp.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Registry.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

// convert id of our binop to id of imex::ndarray binop
static ::imex::ndarray::EWBinOpId sharpy2mlir(const EWBinOpId bop) {
  switch (bop) {
  case __ADD__:
  case ADD:
  case __RADD__:
    return ::imex::ndarray::ADD;
  case ATAN2:
    return ::imex::ndarray::ATAN2;
  case __FLOORDIV__:
  case FLOOR_DIVIDE:
  case __RFLOORDIV__:
    return ::imex::ndarray::FLOOR_DIVIDE;
    // __MATMUL__ is handled before dispatching, see below
  case __MUL__:
  case MULTIPLY:
  case __RMUL__:
    return ::imex::ndarray::MULTIPLY;
  case __SUB__:
  case SUBTRACT:
  case __RSUB__:
    return ::imex::ndarray::SUBTRACT;
  case __TRUEDIV__:
  case DIVIDE:
  case __RTRUEDIV__:
    return ::imex::ndarray::TRUE_DIVIDE;
  case __POW__:
  case POW:
  case __RPOW__:
    return ::imex::ndarray::POWER;
  case LOGADDEXP:
    return ::imex::ndarray::LOGADDEXP;
  case __LSHIFT__:
  case BITWISE_LEFT_SHIFT:
  case __RLSHIFT__:
    return ::imex::ndarray::BITWISE_LEFT_SHIFT;
  case __MOD__:
  case REMAINDER:
  case __RMOD__:
    return ::imex::ndarray::MODULO;
  case __RSHIFT__:
  case BITWISE_RIGHT_SHIFT:
  case __RRSHIFT__:
    return ::imex::ndarray::BITWISE_RIGHT_SHIFT;
  case __AND__:
  case BITWISE_AND:
  case __RAND__:
    return ::imex::ndarray::BITWISE_AND;
  case __OR__:
  case BITWISE_OR:
  case __ROR__:
    return ::imex::ndarray::BITWISE_OR;
  case __XOR__:
  case BITWISE_XOR:
  case __RXOR__:
    return ::imex::ndarray::BITWISE_XOR;
  default:
    throw std::runtime_error("Unknown/invalid elementwise binary operation");
  }
}

bool is_reflected_op(EWBinOpId op) {
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

    auto aTyp = av.getType().cast<::imex::ndarray::NDArrayType>();
    auto outElemType =
        ::imex::ndarray::toMLIR(builder, SHARPY::jit::getPTDType(_dtype));
    auto outTyp = aTyp.cloneWith(shape(), outElemType);

    ::mlir::Value one, two;
    if (is_reflected_op(_op)) {
      one = bv;
      two = av;
    } else {
      one = av;
      two = bv;
    }
    auto bop = builder.create<::imex::ndarray::EWBinOp>(
        loc, outTyp, builder.getI32IntegerAttr(sharpy2mlir(_op)), one, two);

    dm.addVal(
        this->guid(), bop,
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          this->set_value(mk_tnsr(
              this->guid(), _dtype, this->shape(), this->device(), this->team(),
              l_allocated, l_aligned, l_offset, l_sizes, l_strides, o_allocated,
              o_aligned, o_offset, o_sizes, o_strides, r_allocated, r_aligned,
              r_offset, r_sizes, r_strides, std::move(loffs)));
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

FutureArray *EWBinOp::op(EWBinOpId op, const py::object &a,
                         const py::object &b) {
  std::string deva, devb;
  uint64_t teama = 0, teamb = 0;
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
