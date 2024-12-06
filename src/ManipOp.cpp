// SPDX-License-Identifier: BSD-3-Clause

/*
  Manipulation ops.
*/

#include "sharpy/ManipOp.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

struct DeferredReshape : public Deferred {
  enum CopyMode : char { COPY_NEVER, COPY_ALWAYS, COPY_POSSIBLE };
  id_type _a;
  CopyMode _copy;

  DeferredReshape() = default;
  DeferredReshape(const array_i::future_type &a, const shape_type &shape,
                  CopyMode copy)
      : Deferred(a.dtype(), shape, a.device(), a.team()), _a(a.guid()),
        _copy(copy) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, Registry::get(_a));
    ::mlir::SmallVector<::mlir::Value> shp(shape().size());
    for (auto i = 0ul; i < shape().size(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, shape()[i]);
    }
    auto copyA =
        _copy == COPY_POSSIBLE
            ? ::mlir::IntegerAttr()
            : ::imex::getIntAttr(builder, COPY_ALWAYS ? true : false, 1);

    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(av.getType());
    auto outTyp = ::mlir::cast<::mlir::RankedTensorType>(
        aTyp.cloneWith(shape(), aTyp.getElementType()));

    auto op =
        builder.create<::imex::ndarray::ReshapeOp>(loc, outTyp, av, shp, copyA);

    dm.addVal(this->guid(), op,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                auto t =
                    mk_tnsr(this->guid(), _dtype, this->shape(), this->device(),
                            this->team(), allocated, aligned, offset, sizes,
                            strides, std::move(loffs));
                if (_copy != COPY_ALWAYS) {
                  throw std::runtime_error("copy-free reshape not supported");
                  if (Registry::has(_a)) {
                    t->set_base(Registry::get(_a).get());
                  } // else _a is a temporary and was dropped
                }
                this->set_value(std::move(t));
              });

    return false;
  }

  FactoryId factory() const override { return F_RESHAPE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    // ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_copy)>(_copy);
  }
};

// ***************************************************************************

struct DeferredAsType : public Deferred {
  id_type _a;
  bool _copy;

  DeferredAsType() = default;
  DeferredAsType(const array_i::future_type &a, DTypeId dtype, bool copy)
      : Deferred(dtype, a.shape(), a.device(), a.team()), _a(a.guid()),
        _copy(copy) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, Registry::get(_a));
    auto arType = ::mlir::dyn_cast<::mlir::RankedTensorType>(av.getType());
    if (!arType) {
      throw std::invalid_argument("Encountered unexpected type in astype.");
    }

    // construct NDArrayType with same shape and given dtype
    auto mlirElType = jit::getMLIRType(builder, this->dtype());
    auto outType = ::mlir::cast<::mlir::RankedTensorType>(
        arType.cloneWith(std::nullopt, mlirElType));
    auto res = builder.create<::imex::ndarray::CastElemTypeOp>(
        loc, outType, av, ::imex::getIntAttr(builder, _copy, 1));
    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                auto t =
                    mk_tnsr(this->guid(), this->dtype(), this->shape(),
                            this->device(), this->team(), allocated, aligned,
                            offset, sizes, strides, std::move(loffs));
                if (!this->_copy && Registry::has(_a)) {
                  t->set_base(Registry::get(_a).get());
                } // else _a is a temporary and was dropped
                this->set_value(std::move(t));
              });
    return false;
  }

  FactoryId factory() const override { return F_ASTYPE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_copy)>(_copy);
  }
};

// ***************************************************************************

struct DeferredToDevice : public Deferred {
  id_type _a;

  DeferredToDevice() = default;
  DeferredToDevice(const array_i::future_type &a, const std::string &device)
      : Deferred(a.dtype(), a.shape(), device, a.team()), _a(a.guid()) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, Registry::get(_a));
    auto srcType = ::mlir::dyn_cast<::mlir::RankedTensorType>(av.getType());
    if (!srcType) {
      throw std::invalid_argument("Encountered unexpected type in to_device.");
    }

    // copy envs, drop gpu env (if any)
    ::mlir::SmallVector<::mlir::Attribute> envs;
    if (auto srcEnvs = srcType.getEncoding()) {
      auto casted = mlir::cast<::imex::ndarray::EnvsAttr>(srcEnvs);
      for (auto e : casted.getEnvs()) {
        if (!::mlir::isa<::imex::region::GPUEnvAttr>(e)) {
          envs.emplace_back(e);
        }
      }
    }
    // append device attr
    if (!_device.empty()) {
      envs.emplace_back(
          ::imex::region::GPUEnvAttr::get(builder.getStringAttr(_device)));
    }
    auto envsAttr = ::imex::ndarray::EnvsAttr::get(builder.getContext(), envs);
    auto outType = mlir::RankedTensorType::get(
        srcType.getShape(), srcType.getElementType(), envsAttr);
    auto res = builder.create<::imex::ndarray::CopyOp>(loc, outType, av);
    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                auto t =
                    mk_tnsr(this->guid(), this->dtype(), this->shape(),
                            this->device(), this->team(), allocated, aligned,
                            offset, sizes, strides, std::move(loffs));
                this->set_value(std::move(t));
              });
    return false;
  }

  FactoryId factory() const override { return F_TODEVICE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
  }
};

struct DeferredPermuteDims : public Deferred {
  id_type _array;
  shape_type _axes;

  DeferredPermuteDims() = default;
  DeferredPermuteDims(const array_i::future_type &array,
                      const shape_type &shape, const shape_type &axes)
      : Deferred(array.dtype(), shape, array.device(), array.team()),
        _array(array.guid()), _axes(axes) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto arrayValue = dm.getDependent(builder, Registry::get(_array));
    auto aTyp = ::mlir::cast<::mlir::RankedTensorType>(arrayValue.getType());
    mlir::Value out = builder.create<mlir::tensor::EmptyOp>(
        loc, shape(), aTyp.getElementType());
    auto res =
        builder.create<mlir::linalg::TransposeOp>(loc, arrayValue, out, _axes)
            ->getResult(0);

    dm.addVal(this->guid(), res,
              [this](uint64_t rank, void *allocated, void *aligned,
                     intptr_t offset, const intptr_t *sizes,
                     const intptr_t *strides, std::vector<int64_t> &&loffs) {
                auto t =
                    mk_tnsr(this->guid(), _dtype, this->shape(), this->device(),
                            this->team(), allocated, aligned, offset, sizes,
                            strides, std::move(loffs));
                this->set_value(std::move(t));
              });

    return false;
  }

  FactoryId factory() const override { return F_PERMUTEDIMS; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_array)>(_array);
  }
};

FutureArray *ManipOp::reshape(const FutureArray &a, const shape_type &shape,
                              const py::object &copy) {
  auto doCopy = copy.is_none()
                    ? DeferredReshape::COPY_POSSIBLE
                    : (copy.cast<bool>() ? DeferredReshape::COPY_ALWAYS
                                         : DeferredReshape::COPY_NEVER);
  if (doCopy == DeferredReshape::COPY_NEVER) {
    throw std::runtime_error("zero-copy reshape not supported");
  }
  doCopy = DeferredReshape::COPY_ALWAYS;
  return new FutureArray(defer<DeferredReshape>(a.get(), shape, doCopy));
}

FutureArray *ManipOp::astype(const FutureArray &a, DTypeId dtype,
                             const py::object &copy) {
  auto doCopy = copy.is_none() ? false : copy.cast<bool>();
  return new FutureArray(defer<DeferredAsType>(a.get(), dtype, doCopy));
}

FutureArray *ManipOp::to_device(const FutureArray &a,
                                const std::string &device) {
  return new FutureArray(defer<DeferredToDevice>(a.get(), device));
}

FutureArray *ManipOp::permute_dims(const FutureArray &array,
                                   const shape_type &axes) {
  auto shape = array.get().shape();

  // verifyPermuteArray
  if (shape.size() != axes.size()) {
    throw std::invalid_argument("axes must have the same length as the shape");
  }
  for (auto i = 0ul; i < shape.size(); ++i) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      throw std::invalid_argument("axes must contain all dimensions");
    }
  }

  auto permutedShape = shape_type(shape.size());
  for (auto i = 0ul; i < shape.size(); ++i) {
    permutedShape[i] = shape[axes[i]];
  }

  return new FutureArray(
      defer<DeferredPermuteDims>(array.get(), permutedShape, axes));
}

FACTORY_INIT(DeferredReshape, F_RESHAPE);
FACTORY_INIT(DeferredAsType, F_ASTYPE);
FACTORY_INIT(DeferredToDevice, F_TODEVICE);
FACTORY_INIT(DeferredPermuteDims, F_PERMUTEDIMS);

} // namespace SHARPY
