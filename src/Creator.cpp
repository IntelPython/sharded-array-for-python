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

#if 0
namespace x {

    template<typename T>
    class Creator
    {
    public:
        using ptr_type = typename tensor_i::ptr_type;
        using typed_ptr_type = typename DPTensorX<T>::typed_ptr_type;

        static ptr_type op(CreatorId c, const shape_type & shp)
        {
            PVSlice pvslice(shp);
            shape_type shape(std::move(pvslice.tile_shape()));
            switch(c) {
            case EMPTY:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::empty<T>(std::move(shape))));
            case ONES:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::ones<T>(std::move(shape))));
            case ZEROS:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::zeros<T>(std::move(shape))));
            default:
                throw std::runtime_error("Unknown creator");
            };
        };

        static ptr_type op(CreatorId c, const shape_type & shp, PyScalar v)
        {
            T val;
            if constexpr (std::is_integral<T>::value) val = static_cast<T>(v._int);
            else if constexpr (std::is_floating_point<T>::value) val = static_cast<T>(v._float);
            if(c == FULL) {
                if(VPROD(shp) <= 1) {
                    return operatorx<T>::mk_tx(val, REPLICATED);
                }
                PVSlice pvslice(shp);
                shape_type shape(std::move(pvslice.tile_shape()));
                auto a = xt::empty<T>(std::move(shape));
                a.fill(val);
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
            }
            throw std::runtime_error("Unknown creator");
        }

        static ptr_type op(uint64_t start, uint64_t end, uint64_t step)
        {
            PVSlice pvslice({static_cast<uint64_t>(Slice(start, end, step).size())});
            auto lslc = pvslice.local_slice();
            const auto & l1dslc = lslc.dim(0);

            auto a = xt::arange<T>(start + l1dslc._start*step, start + l1dslc._end * step, l1dslc._step);
            auto r = operatorx<T>::mk_tx(std::move(pvslice), std::move(a));

            return r;
        }
    }; // class creatorx
} // namespace x
#endif // if 0

struct DeferredFromShape : public Deferred {
  shape_type _shape;
  DTypeId _dtype;
  CreatorId _op;

  DeferredFromShape() = default;
  DeferredFromShape(CreatorId op, const shape_type &shape, DTypeId dtype)
      : Deferred(dtype, shape.size(), true), _shape(shape), _dtype(dtype),
        _op(op) {}

  void run() {
    // set_value(std::move(TypeDispatch<x::Creator>(_dtype, _op, _shape)));
  }

  // FIXME mlir

  FactoryId factory() const { return F_FROMSHAPE; }

  template <typename S> void serialize(S &ser) {
    ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_dtype)>(_dtype);
    ser.template value<sizeof(_op)>(_op);
  }
};

ddptensor *Creator::create_from_shape(CreatorId op, const shape_type &shape,
                                      DTypeId dtype) {
  return new ddptensor(defer<DeferredFromShape>(op, shape, dtype));
}

struct DeferredFull : public Deferred {
  shape_type _shape;
  PyScalar _val;
  DTypeId _dtype;

  DeferredFull() = default;
  DeferredFull(const shape_type &shape, PyScalar val, DTypeId dtype)
      : Deferred(dtype, shape.size(), true), _shape(shape), _val(val),
        _dtype(dtype) {}

  void run() {
    // auto op = FULL;
    // set_value(std::move(TypeDispatch<x::Creator>(_dtype, op, _shape, _val)));
  }

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
    ::mlir::SmallVector<::mlir::Value> shp(_shape.size());
    for (auto i = 0; i < _shape.size(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, _shape[i]);
    }

    ::imex::ptensor::DType dtyp;
    ::mlir::Value val = dispatch<ValAndDType>(_dtype, builder, loc, _val, dtyp);

    auto team = ::imex::createIndex(
        loc, builder, reinterpret_cast<uint64_t>(getTransceiver()));

    dm.addVal(this->guid(),
              builder.create<::imex::ptensor::CreateOp>(loc, shp, dtyp, val,
                                                        nullptr, team),
              [this](Transceiver *transceiver, uint64_t rank, void *allocated,
                     void *aligned, intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, uint64_t *gs_allocated,
                     uint64_t *gs_aligned, uint64_t *lo_allocated,
                     uint64_t *lo_aligned, uint64_t balanced) {
                assert(rank == this->_shape.size());
                this->set_value(std::move(
                    mk_tnsr(transceiver, _dtype, rank, allocated, aligned,
                            offset, sizes, strides, gs_allocated, gs_aligned,
                            lo_allocated, lo_aligned, balanced)));
              });
    return false;
  }

  FactoryId factory() const { return F_FULL; }

  template <typename S> void serialize(S &ser) {
    ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_val)>(_val._int);
    ser.template value<sizeof(_dtype)>(_dtype);
  }
};

ddptensor *Creator::full(const shape_type &shape, const py::object &val,
                         DTypeId dtype) {
  auto v = mk_scalar(val, dtype);
  return new ddptensor(defer<DeferredFull>(shape, v, dtype));
}

struct DeferredArange : public Deferred {
  uint64_t _start, _end, _step, _team;

  DeferredArange() = default;
  DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype,
                 uint64_t team = 0)
      : Deferred(dtype, 1, true), _start(start), _end(end), _step(step),
        _team(team) {}

  void run() override{
      // set_value(std::move(TypeDispatch<x::Creator>(_dtype, _start, _end,
      // _step)));
  };

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    auto start = ::imex::createInt(loc, builder, _start);
    auto stop = ::imex::createInt(loc, builder, _end);
    auto step = ::imex::createInt(loc, builder, _step);
    // ::mlir::Value
    auto team = ::imex::createIndex(
        loc, builder, reinterpret_cast<uint64_t>(getTransceiver()));
    dm.addVal(this->guid(),
              builder.create<::imex::ptensor::ARangeOp>(loc, start, stop, step,
                                                        nullptr, team),
              [this](Transceiver *transceiver, uint64_t rank, void *allocated,
                     void *aligned, intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, uint64_t *gs_allocated,
                     uint64_t *gs_aligned, uint64_t *lo_allocated,
                     uint64_t *lo_aligned, uint64_t balanced) {
                assert(rank == 1);
                assert(strides[0] == 1);
                this->set_value(std::move(
                    mk_tnsr(transceiver, _dtype, rank, allocated, aligned,
                            offset, sizes, strides, gs_allocated, gs_aligned,
                            lo_allocated, lo_aligned, balanced)));
              });
    return false;
  }

  FactoryId factory() const { return F_ARANGE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_start)>(_start);
    ser.template value<sizeof(_end)>(_end);
    ser.template value<sizeof(_step)>(_step);
    ser.template value<sizeof(_dtype)>(_dtype);
  }
};

ddptensor *Creator::arange(uint64_t start, uint64_t end, uint64_t step,
                           DTypeId dtype, uint64_t team) {
  return new ddptensor(defer<DeferredArange>(start, end, step, dtype, team));
}

std::pair<ddptensor *, bool> Creator::mk_future(const py::object &b) {
  if (py::isinstance<ddptensor>(b)) {
    return {b.cast<ddptensor *>(), false};
  } else if (py::isinstance<py::float_>(b)) {
    return {Creator::full({}, b, FLOAT64), true};
  } else if (py::isinstance<py::int_>(b)) {
    return {Creator::full({}, b, INT64), true};
  }
  throw std::runtime_error(
      "Invalid right operand to elementwise binary operation");
};

FACTORY_INIT(DeferredFromShape, F_FROMSHAPE);
FACTORY_INIT(DeferredFull, F_FULL);
FACTORY_INIT(DeferredArange, F_ARANGE);
