/*
  C++ representation of the array-API's creation functions.
*/

#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Transceiver.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/DepManager.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

static bool FORCE_DIST = get_bool_env("SHARPY_FORCE_DIST");

inline const std::string &mkTeam(const std::string &team) {
  if (!team.empty() && (FORCE_DIST || getTransceiver()->nranks() > 1)) {
    return team;
  }
  static std::string none;
  return none;
}

// check that shape elements are non-negative
void validateShape(const shape_type &shape) {
  for (auto &v : shape) {
    if (v < 0) {
      throw std::invalid_argument(
          "invalid shape, negative dimensions are not allowed\n");
    }
  }
}

imex::ndarray::EnvsAttr mkEnvs(::mlir::Builder &builder, int64_t rank,
                               const std::string &device) {
  if (device.empty()) {
    return nullptr;
  }
  return imex::ndarray::EnvsAttr::get(
      builder.getContext(),
      {::imex::region::GPUEnvAttr::get(builder.getStringAttr(device))});
}

struct DeferredFull : public Deferred {
  PyScalar _val;

  DeferredFull() = default;
  DeferredFull(const shape_type &shape, PyScalar val, DTypeId dtype,
               const std::string &device, const std::string &team)
      : Deferred(dtype, shape, device, team), _val(val) {
    validateShape(shape);
  }

  template <typename T> struct ValAndDType {
    static ::mlir::Value op(::mlir::OpBuilder &builder,
                            const ::mlir::Location &loc, const PyScalar &val,
                            ::mlir::Type &dtyp) {
      dtyp = jit::getMLIRType(builder, DTYPE<T>::value);

      if (is_none(val)) {
        return {};
      } else if constexpr (std::is_floating_point_v<T>) {
        return ::imex::createFloat(loc, builder, val._float, sizeof(T) * 8);
      } else if constexpr (std::is_same_v<bool, T>) {
        return ::imex::createInt(loc, builder, val._int, 1);
      } else if constexpr (std::is_integral_v<T>) {
        return ::imex::createInt(loc, builder, val._int, sizeof(T) * 8);
      }
      throw std::invalid_argument("Unsupported dtype in dispatch");
      return {};
    };
  };

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {

    mlir::Type dtyp;
    ::mlir::Value val = dispatch<ValAndDType>(_dtype, builder, loc, _val, dtyp);
    auto envs = mkEnvs(builder, rank(), _device);
    mlir::Value res =
        builder.create<::mlir::tensor::EmptyOp>(loc, shape(), dtyp, envs);
    if (val) {
      res = builder
                .create<mlir::linalg::FillOp>(loc, mlir::ValueRange{val},
                                              mlir::ValueRange{res})
                .getResult(0);
    }
    res = jit::shardNow(builder, loc, res, team());

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                assert(rank == this->rank());
                this->set_value(mk_tnsr(this->guid(), _dtype, this->shape(),
                                        this->device(), this->team(), allocated,
                                        aligned, offset, sizes, strides,
                                        std::move(loffs)));
              });
    return false;
  }

  FactoryId factory() const override { return F_FULL; }

  template <typename S> void serialize(S &ser) {
    // ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_val)>(_val._int);
    ser.template value<sizeof(_dtype)>(_dtype);
  }
};

FutureArray *Creator::full(const shape_type &shape, const py::object &val,
                           DTypeId dtype, const std::string &device,
                           const std::string &team) {
  auto v = mk_scalar(val, dtype);
  return new FutureArray(
      defer<DeferredFull>(shape, v, dtype, device, mkTeam(team)));
}

// ***************************************************************************

struct DeferredArange : public Deferred {
  uint64_t _start, _end, _step;

  DeferredArange() = default;
  DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype,
                 const std::string &device, const std::string &team)
      : Deferred(dtype,
                 {static_cast<shape_type::value_type>(
                     (end - start + step + (step < 0 ? 1 : -1)) / step)},
                 device, team),
        _start(start), _end(end), _step(step) {
    if (_start > _end && _step > -1ul) {
      throw std::invalid_argument("start > end and step > -1 in arange");
    }
    if (_start < _end && _step < 1) {
      throw std::invalid_argument("start < end and step < 1 in arange");
    }
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto _num = shape()[0];
    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _start + _num * _step);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto dtyp = jit::getMLIRType(builder, dtype());
    auto envs = mkEnvs(builder, rank(), _device);
    auto outType = mlir::RankedTensorType::get(shape(), dtyp, envs);
    mlir::Value res = builder.create<::imex::ndarray::LinSpaceOp>(
        loc, outType, start, stop, num, false);
    res = jit::shardNow(builder, loc, res, team());

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                assert(rank == 1);
                assert(strides[0] == 1);
                this->set_value(mk_tnsr(this->guid(), _dtype, this->shape(),
                                        this->device(), this->team(), allocated,
                                        aligned, offset, sizes, strides,
                                        std::move(loffs)));
              });
    return false;
  }

  FactoryId factory() const override { return F_ARANGE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_start)>(_start);
    ser.template value<sizeof(_end)>(_end);
    ser.template value<sizeof(_step)>(_step);
  }
};

FutureArray *Creator::arange(uint64_t start, uint64_t end, uint64_t step,
                             DTypeId dtype, const std::string &device,
                             const std::string &team) {
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
                   DTypeId dtype, const std::string &device,
                   const std::string &team)
      : Deferred(dtype, {static_cast<shape_type::value_type>(num)}, device,
                 team),
        _start(start), _end(end), _num(num), _endpoint(endpoint) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto start = ::imex::createFloat(loc, builder, _start);
    auto stop = ::imex::createFloat(loc, builder, _end);
    auto num = ::imex::createIndex(loc, builder, _num);
    auto dtyp = jit::getMLIRType(builder, dtype());
    auto envs = mkEnvs(builder, rank(), _device);
    auto outType = mlir::RankedTensorType::get(shape(), dtyp, envs);
    mlir::Value res = builder.create<::imex::ndarray::LinSpaceOp>(
        loc, outType, start, stop, num, _endpoint);
    res = jit::shardNow(builder, loc, res, team());

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                assert(rank == 1);
                assert(strides[0] == 1);
                this->set_value(mk_tnsr(this->guid(), _dtype, this->shape(),
                                        this->device(), this->team(), allocated,
                                        aligned, offset, sizes, strides,
                                        std::move(loffs)));
              });
    return false;
  }

  FactoryId factory() const override { return F_ARANGE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_start)>(_start);
    ser.template value<sizeof(_end)>(_end);
    ser.template value<sizeof(_num)>(_num);
    ser.template value<sizeof(_endpoint)>(_endpoint);
  }
};

FutureArray *Creator::linspace(double start, double end, uint64_t num,
                               bool endpoint, DTypeId dtype,
                               const std::string &device,
                               const std::string &team) {
  return new FutureArray(defer<DeferredLinspace>(start, end, num, endpoint,
                                                 dtype, device, mkTeam(team)));
}

// ***************************************************************************

extern DTypeId DEFAULT_FLOAT;
extern DTypeId DEFAULT_INT;

std::pair<FutureArray *, bool> Creator::mk_future(const py::object &b,
                                                  const std::string &device,
                                                  const std::string &team,
                                                  DTypeId dtype) {
  if (py::isinstance<FutureArray>(b)) {
    return {b.cast<FutureArray *>(), false};
  } else if (py::isinstance<py::float_>(b) || py::isinstance<py::int_>(b)) {
    return {Creator::full({}, b, dtype, device, team), true};
  }
  throw std::invalid_argument(
      "Invalid right operand to elementwise binary operation");
};

FACTORY_INIT(DeferredFull, F_FULL);
FACTORY_INIT(DeferredArange, F_ARANGE);
FACTORY_INIT(DeferredLinspace, F_LINSPACE);
} // namespace SHARPY
