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

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
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
    for (auto i = 0; i < shape().size(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, shape()[i]);
    }
    auto copyA =
        _copy == COPY_POSSIBLE
            ? ::mlir::IntegerAttr()
            : ::imex::getIntAttr(builder, COPY_ALWAYS ? true : false, 1);

    auto aTyp = av.getType().cast<::imex::ndarray::NDArrayType>();
    auto outTyp = aTyp.cloneWith(shape(), aTyp.getElementType());

    auto op =
        builder.create<::imex::ndarray::ReshapeOp>(loc, outTyp, av, shp, copyA);

    dm.addVal(
        this->guid(), op,
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          auto t = mk_tnsr(this->guid(), _dtype, this->shape(), this->device(),
                           this->team(), l_allocated, l_aligned, l_offset,
                           l_sizes, l_strides, o_allocated, o_aligned, o_offset,
                           o_sizes, o_strides, r_allocated, r_aligned, r_offset,
                           r_sizes, r_strides, std::move(loffs));
          if (_copy != COPY_ALWAYS) {
            assert(!"copy-free reshape not supported");
            if (Registry::has(_a)) {
              t->set_base(Registry::get(_a).get());
            } // else _a is a temporary and was dropped
          }
          this->set_value(std::move(t));
        });

    return false;
  }

  FactoryId factory() const { return F_RESHAPE; }

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

  template <typename T> struct convDType {
    static ::imex::ndarray::DType op() { return jit::PT_DTYPE<T>::value; };
  };

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    const auto dtype = this->dtype();
    auto av = dm.getDependent(builder, Registry::get(_a));

    auto copyAttr = ::imex::getIntAttr(builder, _copy, 1);
    // construct NDArrayType with same shape and given dtype
    ::imex::ndarray::DType ndDType = dispatch<convDType>(dtype);
    auto mlirElType = ::imex::ndarray::toMLIR(builder, ndDType);
    auto arType = av.getType().dyn_cast<::imex::ndarray::NDArrayType>();
    assert(arType);
    auto outType = arType.cloneWith(std::nullopt, mlirElType);
    auto res = builder.create<::imex::ndarray::CastElemTypeOp>(loc, outType, av,
                                                               copyAttr);
    dm.addVal(
        this->guid(), res,
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          auto t = mk_tnsr(this->guid(), this->dtype(), this->shape(),
                           this->device(), this->team(), l_allocated, l_aligned,
                           l_offset, l_sizes, l_strides, o_allocated, o_aligned,
                           o_offset, o_sizes, o_strides, r_allocated, r_aligned,
                           r_offset, r_sizes, r_strides, std::move(loffs));
          if (!this->_copy && Registry::has(_a)) {
            t->set_base(Registry::get(_a).get());
          } // else _a is a temporary and was dropped
          this->set_value(std::move(t));
        });
    return false;
  }

  FactoryId factory() const { return F_ASTYPE; }

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

    auto srcType = av.getType().dyn_cast<::imex::ndarray::NDArrayType>();
    assert(srcType);
    // copy envs, drop gpu env (if any)
    auto srcEnvs = srcType.getEnvironments();
    ::mlir::SmallVector<::mlir::Attribute> envs;
    for (auto e : srcEnvs) {
      if (!::mlir::isa<::imex::region::GPUEnvAttr>(e)) {
        envs.emplace_back(e);
      }
    }
    // append device attr
    if (!_device.empty()) {
      envs.emplace_back(
          ::imex::region::GPUEnvAttr::get(builder.getStringAttr(_device)));
    }
    auto outType = ::imex::ndarray::NDArrayType::get(srcType.getShape(),
                                                     srcType.getElementType(),
                                                     envs, srcType.getLayout());
    auto res = builder.create<::imex::ndarray::CopyOp>(loc, outType, av);
    dm.addVal(
        this->guid(), res,
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          auto t = mk_tnsr(this->guid(), this->dtype(), this->shape(),
                           this->device(), this->team(), l_allocated, l_aligned,
                           l_offset, l_sizes, l_strides, o_allocated, o_aligned,
                           o_offset, o_sizes, o_strides, r_allocated, r_aligned,
                           r_offset, r_sizes, r_strides, std::move(loffs));
          this->set_value(std::move(t));
        });
    return false;
  }

  FactoryId factory() const { return F_TODEVICE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
  }
};

FutureArray *ManipOp::reshape(const FutureArray &a, const shape_type &shape,
                              const py::object &copy) {
  auto doCopy = copy.is_none()
                    ? DeferredReshape::COPY_POSSIBLE
                    : (copy.cast<bool>() ? DeferredReshape::COPY_ALWAYS
                                         : DeferredReshape::COPY_NEVER);
  if (doCopy == DeferredReshape::COPY_NEVER) {
    assert(!"zero-copy reshape not supported");
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

FACTORY_INIT(DeferredReshape, F_RESHAPE);
FACTORY_INIT(DeferredAsType, F_ASTYPE);
FACTORY_INIT(DeferredToDevice, F_TODEVICE);
} // namespace SHARPY
