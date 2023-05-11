// SPDX-License-Identifier: BSD-3-Clause

/*
  Manipulation ops.
*/

#include "ddptensor/ManipOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/TypeDispatch.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/IR/Builders.h>

struct DeferredReshape : public Deferred {
  enum CopyMode : char { COPY_NEVER, COPY_ALWAYS, COPY_POSSIBLE };
  id_type _a;
  shape_type _shape;
  CopyMode _copy;

  DeferredReshape() = default;
  DeferredReshape(const tensor_i::future_type &a, const shape_type &shape,
                  CopyMode copy)
      : Deferred(a.dtype(), shape.size(), a.team(), true), _a(a.guid()),
        _shape(shape), _copy(copy) {}

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, _a);
    ::mlir::SmallVector<::mlir::Value> shp(_shape.size());
    for (auto i = 0; i < _shape.size(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, _shape[i]);
    }
    auto copyA =
        _copy == COPY_POSSIBLE
            ? ::mlir::IntegerAttr()
            : ::imex::getIntAttr<1>(builder, COPY_ALWAYS ? true : false);

    ::mlir::SmallVector<int64_t> nshape(shp.size(),
                                        ::mlir::ShapedType::kDynamic);
    auto outTyp = ::imex::ptensor::PTensorType::get(
        nshape, ::imex::dist::getPTensorType(av).getElementType());
    auto op =
        builder.create<::imex::ptensor::ReshapeOp>(loc, outTyp, av, shp, copyA);

    auto future_a = Registry::get(_a);

    dm.addVal(this->guid(), op,
              [this, future_a](Transceiver *transceiver, uint64_t rank,
                               void *allocated, void *aligned, intptr_t offset,
                               const intptr_t *sizes, const intptr_t *strides,
                               int64_t *gs_allocated, int64_t *gs_aligned,
                               uint64_t *lo_allocated, uint64_t *lo_aligned,
                               uint64_t balanced) {
                auto t =
                    mk_tnsr(transceiver, _dtype, rank, allocated, aligned,
                            offset, sizes, strides, gs_allocated, gs_aligned,
                            lo_allocated, lo_aligned, balanced);
                if (_copy != COPY_ALWAYS) {
                  assert(!"copy-free reshape not supported");
                  t->set_base(future_a.get());
                }
                this->set_value(std::move(t));
              });

    return false;
  }

  FactoryId factory() const { return F_RESHAPE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_copy)>(_copy);
  }
};

ddptensor *ManipOp::reshape(const ddptensor &a, const shape_type &shape,
                            const py::object &copy) {
  auto doCopy = copy.is_none()
                    ? DeferredReshape::COPY_POSSIBLE
                    : (copy.cast<bool>() ? DeferredReshape::COPY_ALWAYS
                                         : DeferredReshape::COPY_NEVER);
  if (doCopy == DeferredReshape::COPY_NEVER) {
    assert(!"zero-copy reshape not supported");
  }
  doCopy = DeferredReshape::COPY_ALWAYS;
  return new ddptensor(defer<DeferredReshape>(a.get(), shape, doCopy));
}

FACTORY_INIT(DeferredReshape, F_RESHAPE);
