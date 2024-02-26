/*
  C++ representation of the array-API's creation functions.
*/

#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Transceiver.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

static const char *FORCE_DIST = getenv("SHARPY_FORCE_DIST");

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
               const std::string &device, uint64_t team)
      : Deferred(dtype, shape, device, team), _val(val) {}

  template <typename T> struct ValAndDType {
    static ::mlir::Value op(::mlir::OpBuilder &builder,
                            const ::mlir::Location &loc, const PyScalar &val,
                            ::imex::ndarray::DType &dtyp) {
      dtyp = jit::PT_DTYPE<T>::value;

      if (is_none(val)) {
        return {};
      } else if constexpr (std::is_floating_point_v<T>) {
        return ::imex::createFloat(loc, builder, val._float, sizeof(T) * 8);
      } else if constexpr (std::is_same_v<bool, T>) {
        return ::imex::createInt(loc, builder, val._int, 1);
      } else if constexpr (std::is_integral_v<T>) {
        return ::imex::createInt(loc, builder, val._int, sizeof(T) * 8);
      }
      assert("Unsupported dtype in dispatch");
      return {};
    };
  };

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    ::mlir::SmallVector<::mlir::Value> shp(rank());
    for (auto i = 0ul; i < rank(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, shape()[i]);
    }

    ::imex::ndarray::DType dtyp;
    ::mlir::Value val = dispatch<ValAndDType>(_dtype, builder, loc, _val, dtyp);
    auto envs = jit::mkEnvs(builder, rank(), _device, team());

    dm.addVal(
        this->guid(),
        builder.create<::imex::ndarray::CreateOp>(loc, shp, dtyp, val, envs),
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          assert(rank == this->rank());
          this->set_value(std::move(mk_tnsr(
              this->guid(), _dtype, this->shape(), this->device(), this->team(),
              l_allocated, l_aligned, l_offset, l_sizes, l_strides, o_allocated,
              o_aligned, o_offset, o_sizes, o_strides, r_allocated, r_aligned,
              r_offset, r_sizes, r_strides, std::move(loffs))));
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

FutureArray *Creator::full(const shape_type &shape, const py::object &val,
                           DTypeId dtype, const std::string &device,
                           uint64_t team) {
  auto v = mk_scalar(val, dtype);
  return new FutureArray(
      defer<DeferredFull>(shape, v, dtype, device, mkTeam(team)));
}

// ***************************************************************************

struct DeferredArange : public Deferred {
  uint64_t _start, _end, _step;

  DeferredArange() = default;
  DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype,
                 const std::string &device, uint64_t team)
      : Deferred(dtype,
                 {static_cast<shape_type::value_type>(
                     (end - start + step + (step < 0 ? 1 : -1)) / step)},
                 device, team),
        _start(start), _end(end), _step(step) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto _num = shape()[0];
    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _start + _num * _step);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto dtyp = jit::getPTDType(dtype());
    auto envs = jit::mkEnvs(builder, rank(), _device, team());

    dm.addVal(
        this->guid(),
        builder.create<::imex::ndarray::LinSpaceOp>(loc, start, stop, num,
                                                    false, dtyp, envs),
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          assert(rank == 1);
          assert(o_strides[0] == 1);
          this->set_value(std::move(mk_tnsr(
              this->guid(), _dtype, this->shape(), this->device(), this->team(),
              l_allocated, l_aligned, l_offset, l_sizes, l_strides, o_allocated,
              o_aligned, o_offset, o_sizes, o_strides, r_allocated, r_aligned,
              r_offset, r_sizes, r_strides, std::move(loffs))));
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

FutureArray *Creator::arange(uint64_t start, uint64_t end, uint64_t step,
                             DTypeId dtype, const std::string &device,
                             uint64_t team) {
  return new FutureArray(
      defer<DeferredArange>(start, end, step, dtype, device, mkTeam(team)));
}

// ***************************************************************************

struct DeferredLinspace : public Deferred {
  double _start, _end;
  uint64_t _num;
  bool _endpoint;

  DeferredLinspace() = default;
  DeferredLinspace(double start, double end, uint64_t num, bool endpoint,
                   DTypeId dtype, const std::string &device, uint64_t team)
      : Deferred(dtype, {static_cast<shape_type::value_type>(num)}, device,
                 team),
        _start(start), _end(end), _num(num), _endpoint(endpoint) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _end);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto dtyp = jit::getPTDType(dtype());
    auto envs = jit::mkEnvs(builder, rank(), _device, team());

    dm.addVal(
        this->guid(),
        builder.create<::imex::ndarray::LinSpaceOp>(loc, start, stop, num,
                                                    _endpoint, dtyp, envs),
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          assert(rank == 1);
          assert(l_strides[0] == 1);
          this->set_value(std::move(mk_tnsr(
              this->guid(), _dtype, this->shape(), this->device(), this->team(),
              l_allocated, l_aligned, l_offset, l_sizes, l_strides, o_allocated,
              o_aligned, o_offset, o_sizes, o_strides, r_allocated, r_aligned,
              r_offset, r_sizes, r_strides, std::move(loffs))));
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

FutureArray *Creator::linspace(double start, double end, uint64_t num,
                               bool endpoint, DTypeId dtype,
                               const std::string &device, uint64_t team) {
  return new FutureArray(defer<DeferredLinspace>(start, end, num, endpoint,
                                                 dtype, device, mkTeam(team)));
}

// ***************************************************************************

extern DTypeId DEFAULT_FLOAT;
extern DTypeId DEFAULT_INT;

std::pair<FutureArray *, bool> Creator::mk_future(const py::object &b,
                                                  const std::string &device,
                                                  uint64_t team,
                                                  DTypeId dtype) {
  if (py::isinstance<FutureArray>(b)) {
    return {b.cast<FutureArray *>(), false};
  } else if (py::isinstance<py::float_>(b) || py::isinstance<py::int_>(b)) {
    return {Creator::full({}, b, dtype, device, team), true};
  }
  throw std::runtime_error(
      "Invalid right operand to elementwise binary operation");
};

FACTORY_INIT(DeferredFull, F_FULL);
FACTORY_INIT(DeferredArange, F_ARANGE);
FACTORY_INIT(DeferredLinspace, F_LINSPACE);
} // namespace SHARPY
