/*
  C++ representation of the array-API's creation functions.
*/

#include "ddptensor/Creator.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Transceiver.hpp"
#include "ddptensor/TypeDispatch.hpp"

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>

static const char *FORCE_DIST = getenv("DDPT_FORCE_DIST");

inline uint64_t mkTeam(uint64_t team) {
  if (team && (FORCE_DIST || getTransceiver()->nranks() > 1)) {
    return 1;
  }
  return 0;
}

struct DeferredFull : public Deferred {
  PyScalar _val;

  DeferredFull() = default;
  DeferredFull(const shape_type &shape, PyScalar val, DTypeId dtype,
               uint64_t team)
      : Deferred(dtype, shape, team, true), _val(val) {}

  template <typename T> struct ValAndDType {
    static ::mlir::Value op(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            const PyScalar &val, ::imex::ptensor::DType &dtyp) {
      dtyp = jit::PT_DTYPE<T>::value;

      if (is_none(val)) {
        return {};
      } else if constexpr (std::is_floating_point_v<T>) {
        return ::imex::createFloat<sizeof(T) * 8>(loc, builder, val._float);
      } else if constexpr (std::is_same_v<bool, T>) {
        return ::imex::createInt<1>(loc, builder, val._int);
      } else if constexpr (std::is_integral_v<T>) {
        return ::imex::createInt<sizeof(T) * 8>(loc, builder, val._int);
      }
      assert("Unsupported dtype in dispatch");
      return {};
    };
  };

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    ::mlir::SmallVector<::mlir::Value> shp(rank());
    for (auto i = 0; i < rank(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, shape()[i]);
    }

    ::imex::ptensor::DType dtyp;
    ::mlir::Value val = dispatch<ValAndDType>(_dtype, builder, loc, _val, dtyp);

    auto transceiver = getTransceiver();
    auto teamV = team() == 0
                     ? ::mlir::Value()
                     : ::imex::createIndex(loc, builder,
                                           reinterpret_cast<uint64_t>(team()));

    auto rTyp = ::imex::ptensor::PTensorType::get(
        shape(), imex::ptensor::toMLIR(builder, dtyp));

    dm.addVal(this->guid(),
              builder.create<::imex::ptensor::CreateOp>(loc, rTyp, shp, dtyp,
                                                        val, nullptr, teamV),
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                assert(rank == this->rank());
                this->set_value(std::move(mk_tnsr(
                    reinterpret_cast<Transceiver *>(this->team()), _dtype,
                    this->shape(), l_allocated, l_aligned, l_offset, l_sizes,
                    l_strides, o_allocated, o_aligned, o_offset, o_sizes,
                    o_strides, r_allocated, r_aligned, r_offset, r_sizes,
                    r_strides, lo_allocated, lo_aligned)));
              });
    return false;
  }

  FactoryId factory() const { return F_FULL; }

  template <typename S> void serialize(S &ser) {
    // ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_val)>(_val._int);
    ser.template value<sizeof(_dtype)>(_dtype);
  }
};

ddptensor *Creator::full(const shape_type &shape, const py::object &val,
                         DTypeId dtype, uint64_t team) {
  auto v = mk_scalar(val, dtype);
  return new ddptensor(defer<DeferredFull>(shape, v, dtype, mkTeam(team)));
}

// ***************************************************************************

struct DeferredArange : public Deferred {
  uint64_t _start, _end, _step;

  DeferredArange() = default;
  DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype,
                 uint64_t team)
      : Deferred(dtype,
                 {static_cast<shape_type::value_type>(
                     (end - start + step + (step < 0 ? 1 : -1)) / step)},
                 team, true),
        _start(start), _end(end), _step(step) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // ::mlir::Value
    auto transceiver = getTransceiver();
    auto teamV = team() == 0
                     ? ::mlir::Value()
                     : ::imex::createIndex(loc, builder,
                                           reinterpret_cast<uint64_t>(team()));

    auto _num = shape()[0];

    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _start + _num * _step);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto rTyp = ::imex::ptensor::PTensorType::get(
        shape(), imex::ptensor::toMLIR(builder, jit::getPTDType(_dtype)));

    dm.addVal(this->guid(),
              builder.create<::imex::ptensor::LinSpaceOp>(
                  loc, rTyp, start, stop, num, false, nullptr, teamV),
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                assert(rank == 1);
                assert(l_strides[0] == 1);
                this->set_value(std::move(mk_tnsr(
                    reinterpret_cast<Transceiver *>(this->team()), _dtype,
                    this->shape(), l_allocated, l_aligned, l_offset, l_sizes,
                    l_strides, o_allocated, o_aligned, o_offset, o_sizes,
                    o_strides, r_allocated, r_aligned, r_offset, r_sizes,
                    r_strides, lo_allocated, lo_aligned)));
              });
    return false;
  }

  FactoryId factory() const { return F_ARANGE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_start)>(_start);
    ser.template value<sizeof(_end)>(_end);
    ser.template value<sizeof(_step)>(_step);
  }
};

ddptensor *Creator::arange(uint64_t start, uint64_t end, uint64_t step,
                           DTypeId dtype, uint64_t team) {
  return new ddptensor(
      defer<DeferredArange>(start, end, step, dtype, mkTeam(team)));
}

// ***************************************************************************

struct DeferredLinspace : public Deferred {
  double _start, _end;
  uint64_t _num;
  bool _endpoint;

  DeferredLinspace() = default;
  DeferredLinspace(double start, double end, uint64_t num, bool endpoint,
                   DTypeId dtype, uint64_t team)
      : Deferred(dtype, {static_cast<shape_type::value_type>(num)}, team, true),
        _start(start), _end(end), _num(num), _endpoint(endpoint) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // ::mlir::Value
    auto teamV = team() == 0
                     ? ::mlir::Value()
                     : ::imex::createIndex(loc, builder,
                                           reinterpret_cast<uint64_t>(team()));

    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _end);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto rTyp = ::imex::ptensor::PTensorType::get(
        shape(), imex::ptensor::toMLIR(builder, jit::getPTDType(_dtype)));

    dm.addVal(this->guid(),
              builder.create<::imex::ptensor::LinSpaceOp>(
                  loc, rTyp, start, stop, num, _endpoint, nullptr, teamV),
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                assert(rank == 1);
                assert(l_strides[0] == 1);
                this->set_value(std::move(mk_tnsr(
                    reinterpret_cast<Transceiver *>(this->team()), _dtype,
                    this->shape(), l_allocated, l_aligned, l_offset, l_sizes,
                    l_strides, o_allocated, o_aligned, o_offset, o_sizes,
                    o_strides, r_allocated, r_aligned, r_offset, r_sizes,
                    r_strides, lo_allocated, lo_aligned)));
              });
    return false;
  }

  FactoryId factory() const { return F_ARANGE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_start)>(_start);
    ser.template value<sizeof(_end)>(_end);
    ser.template value<sizeof(_num)>(_num);
    ser.template value<sizeof(_endpoint)>(_endpoint);
  }
};

ddptensor *Creator::linspace(double start, double end, uint64_t num,
                             bool endpoint, DTypeId dtype, uint64_t team) {
  return new ddptensor(
      defer<DeferredLinspace>(start, end, num, endpoint, dtype, mkTeam(team)));
}

// ***************************************************************************

std::pair<ddptensor *, bool> Creator::mk_future(const py::object &b,
                                                uint64_t team) {
  if (py::isinstance<ddptensor>(b)) {
    return {b.cast<ddptensor *>(), false};
  } else if (py::isinstance<py::float_>(b)) {
    return {Creator::full({}, b, FLOAT64, team), true};
  } else if (py::isinstance<py::int_>(b)) {
    return {Creator::full({}, b, INT64, team), true};
  }
  throw std::runtime_error(
      "Invalid right operand to elementwise binary operation");
};

FACTORY_INIT(DeferredFull, F_FULL);
FACTORY_INIT(DeferredArange, F_ARANGE);
FACTORY_INIT(DeferredLinspace, F_LINSPACE);
