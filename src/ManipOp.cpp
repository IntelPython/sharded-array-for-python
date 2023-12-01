// SPDX-License-Identifier: BSD-3-Clause

/*
  Manipulation ops.
*/

#include "ddptensor/ManipOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/IR/Builders.h>

namespace DDPT {

struct DeferredReshape : public Deferred {
  enum CopyMode : char { COPY_NEVER, COPY_ALWAYS, COPY_POSSIBLE };
  id_type _a;
  CopyMode _copy;

  DeferredReshape() = default;
  DeferredReshape(const tensor_i::future_type &a, const shape_type &shape,
                  CopyMode copy)
      : Deferred(a.dtype(), shape, a.device(), a.team()), _a(a.guid()),
        _copy(copy) {}

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    auto av = dm.getDependent(builder, _a);
    ::mlir::SmallVector<::mlir::Value> shp(shape().size());
    for (auto i = 0; i < shape().size(); ++i) {
      shp[i] = ::imex::createIndex(loc, builder, shape()[i]);
    }
    auto copyA =
        _copy == COPY_POSSIBLE
            ? ::mlir::IntegerAttr()
            : ::imex::getIntAttr(builder, COPY_ALWAYS ? true : false, 1);

    auto aTyp = av.getType().cast<::imex::ptensor::PTensorType>();
    auto outTyp = aTyp.cloneWith(shape(), aTyp.getElementType());

    auto op =
        builder.create<::imex::ptensor::ReshapeOp>(loc, outTyp, av, shp, copyA);

    dm.addVal(this->guid(), op,
              [this](uint64_t rank, void *l_allocated, void *l_aligned,
                     intptr_t l_offset, const intptr_t *l_sizes,
                     const intptr_t *l_strides, void *o_allocated,
                     void *o_aligned, intptr_t o_offset,
                     const intptr_t *o_sizes, const intptr_t *o_strides,
                     void *r_allocated, void *r_aligned, intptr_t r_offset,
                     const intptr_t *r_sizes, const intptr_t *r_strides,
                     uint64_t *lo_allocated, uint64_t *lo_aligned) {
                auto t = mk_tnsr(reinterpret_cast<Transceiver *>(this->team()),
                                 _dtype, this->shape(), l_allocated, l_aligned,
                                 l_offset, l_sizes, l_strides, o_allocated,
                                 o_aligned, o_offset, o_sizes, o_strides,
                                 r_allocated, r_aligned, r_offset, r_sizes,
                                 r_strides, lo_allocated, lo_aligned);
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
} // namespace DDPT
