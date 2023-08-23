// SPDX-License-Identifier: BSD-3-Clause

// Implementation of reduction operations

#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>

#if 0
namespace x {

    class ReduceOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename X>
        static ptr_type dist_reduce(ReduceOpId rop, const PVSlice & slice, const dim_vec_type & dims, X && x)
        {
            xt::xarray<typename X::value_type> a = x;
            auto new_shape = reduce_shape(slice.shape(), dims);
            rank_type owner = NOOWNER;
            if(slice.need_reduce(dims)) {
                auto len = VPROD(new_shape);
                getTransceiver()->reduce_all(a.data(), DTYPE<typename X::value_type>::value, len, rop);
                if(len == 1) return operatorx<typename X::value_type>::mk_tx(a.data()[0], REPLICATED);
                owner = REPLICATED;
            }
            return operatorx<typename X::value_type>::mk_tx(std::move(new_shape), a, owner);
        }

        template<typename T>
        static ptr_type op(ReduceOpId rop, const dim_vec_type & dims, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            const auto & ax = a_ptr->xarray();
            if(a_ptr->is_sliced()) {
                const auto & av = xt::strided_view(ax, a_ptr->lslice());
                return do_op(rop, dims, av, a_ptr);
            }
            return do_op(rop, dims, ax, a_ptr);
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename T1, typename T>
        static ptr_type do_op(ReduceOpId rop, const dim_vec_type & dims, const T1 & a, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            switch(rop) {
            case MEAN:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::mean(a, dims));
            case PROD:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::prod(a, dims));
            case SUM:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::sum(a, dims));
            case STD:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::stddev(a, dims));
            case VAR:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::variance(a, dims));
            case MAX:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::amax(a, dims));
            case MIN:
                return dist_reduce(rop, a_ptr->slice(), dims, xt::amin(a, dims));
            default:
                throw std::runtime_error("Unknown reduction operation");
            }
        }

#pragma GCC diagnostic pop

    };
} // namespace x
#endif // if 0

// convert id of our reduction op to id of imex::ptensor reduction op
static ::imex::ptensor::ReduceOpId ddpt2mlir(const ReduceOpId rop) {
  switch (rop) {
  case MEAN:
    return ::imex::ptensor::MEAN;
  case PROD:
    return ::imex::ptensor::PROD;
  case SUM:
    return ::imex::ptensor::SUM;
  case STD:
    return ::imex::ptensor::STD;
  case VAR:
    return ::imex::ptensor::VAR;
  case MAX:
    return ::imex::ptensor::MAX;
  case MIN:
    return ::imex::ptensor::MIN;
  default:
    throw std::runtime_error("Unknown reduction operation");
  }
}

struct DeferredReduceOp : public Deferred {
  id_type _a;
  dim_vec_type _dim;
  ReduceOpId _op;

  DeferredReduceOp() = default;
  DeferredReduceOp(ReduceOpId op, const tensor_i::future_type &a,
                   const dim_vec_type &dim)
      : Deferred(a.dtype(), {}, a.team(), true), // FIXME rank
        _a(a.guid()), _dim(dim), _op(op) {}

  void run() {
#if 0
        const auto a = std::move(Registry::get(_a).get());
        set_value(std::move(TypeDispatch<x::ReduceOp>(a, _op, _dim)));
#endif
  }

  bool generate_mlir(::mlir::OpBuilder &builder, ::mlir::Location loc,
                     jit::DepManager &dm) override {
    // FIXME reduction over individual dimensions is not supported
    auto av = dm.getDependent(builder, _a);
    ::mlir::Type dtype = ::imex::dist::getElementType(av);
    // return type 0d with same dtype as input
    auto retPtTyp = ::imex::ptensor::PTensorType::get(shape(), dtype);
    // reduction op
    auto mop = ddpt2mlir(_op);
    auto op =
        builder.getIntegerAttr(builder.getIntegerType(sizeof(mop) * 8), mop);
    dm.addVal(
        this->guid(),
        builder.create<::imex::ptensor::ReductionOp>(loc, retPtTyp, op, av),
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, uint64_t *lo_allocated,
               uint64_t *lo_aligned) {
          this->set_value(std::move(
              mk_tnsr(reinterpret_cast<Transceiver *>(this->team()), _dtype,
                      this->shape(), l_allocated, l_aligned, l_offset, l_sizes,
                      l_strides, o_allocated, o_aligned, o_offset, o_sizes,
                      o_strides, r_allocated, r_aligned, r_offset, r_sizes,
                      r_strides, lo_allocated, lo_aligned)));
        });
    return false;
  }

  FactoryId factory() const { return F_REDUCEOP; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template container<sizeof(dim_vec_type::value_type)>(_dim, 8);
    ser.template value<sizeof(_op)>(_op);
  }
};

ddptensor *ReduceOp::op(ReduceOpId op, const ddptensor &a,
                        const dim_vec_type &dim) {
  return new ddptensor(defer<DeferredReduceOp>(op, a.get(), dim));
}

FACTORY_INIT(DeferredReduceOp, F_REDUCEOP);
