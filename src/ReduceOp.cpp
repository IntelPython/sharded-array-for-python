// SPDX-License-Identifier: BSD-3-Clause

// Implementation of reduction operations

#include "sharpy/ReduceOp.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/jit/mlir.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/Builders.h>

namespace SHARPY {

#if 0
namespace x {

    class ReduceOp
    {
    public:
        using ptr_type = DNDArrayBaseX::ptr_type;

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
        static ptr_type op(ReduceOpId rop, const dim_vec_type & dims, const std::shared_ptr<DNDArrayX<T>> & a_ptr)
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
        static ptr_type do_op(ReduceOpId rop, const dim_vec_type & dims, const T1 & a, const std::shared_ptr<DNDArrayX<T>> & a_ptr)
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

// convert id of our reduction op to id of imex::ndarray reduction op
static ::imex::ndarray::ReduceOpId sharpy2mlir(const ReduceOpId rop) {
  switch (rop) {
  case MEAN:
    return ::imex::ndarray::MEAN;
  case PROD:
    return ::imex::ndarray::PROD;
  case SUM:
    return ::imex::ndarray::SUM;
  case STD:
    return ::imex::ndarray::STD;
  case VAR:
    return ::imex::ndarray::VAR;
  case MAX:
    return ::imex::ndarray::MAX;
  case MIN:
    return ::imex::ndarray::MIN;
  default:
    throw std::runtime_error("Unknown reduction operation");
  }
}

struct DeferredReduceOp : public Deferred {
  id_type _a;
  dim_vec_type _dim;
  ReduceOpId _op;

  DeferredReduceOp() = default;
  DeferredReduceOp(ReduceOpId op, const array_i::future_type &a,
                   const dim_vec_type &dim)
      : Deferred(a.dtype(), {}, a.device(), a.team()), // FIXME rank
        _a(a.guid()), _dim(dim), _op(op) {}

  void run() {
#if 0
        const auto a = std::move(Registry::get(_a).get());
        set_value(std::move(TypeDispatch<x::ReduceOp>(a, _op, _dim)));
#endif
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    // FIXME reduction over individual dimensions is not supported
    auto av = dm.getDependent(builder, Registry::get(_a));
    ::mlir::Type dtype =
        av.getType().cast<::imex::ndarray::NDArrayType>().getElementType();
    // return type 0d with same dtype as input
    auto aTyp = av.getType().cast<::imex::ndarray::NDArrayType>();
    auto outTyp = ::imex::dist::cloneWithShape(aTyp, shape());
    // reduction op
    auto mop = sharpy2mlir(_op);
    auto op =
        builder.getIntegerAttr(builder.getIntegerType(sizeof(mop) * 8), mop);
    dm.addVal(
        this->guid(),
        builder.create<::imex::ndarray::ReductionOp>(loc, outTyp, op, av),
        [this](uint64_t rank, void *l_allocated, void *l_aligned,
               intptr_t l_offset, const intptr_t *l_sizes,
               const intptr_t *l_strides, void *o_allocated, void *o_aligned,
               intptr_t o_offset, const intptr_t *o_sizes,
               const intptr_t *o_strides, void *r_allocated, void *r_aligned,
               intptr_t r_offset, const intptr_t *r_sizes,
               const intptr_t *r_strides, std::vector<int64_t> &&loffs) {
          this->set_value(std::move(mk_tnsr(
              this->guid(), _dtype, this->shape(), this->device(), this->team(),
              l_allocated, l_aligned, l_offset, l_sizes, l_strides, o_allocated,
              o_aligned, o_offset, o_sizes, o_strides, r_allocated, r_aligned,
              r_offset, r_sizes, r_strides, std::move(loffs))));
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

FutureArray *ReduceOp::op(ReduceOpId op, const FutureArray &a,
                          const dim_vec_type &dim) {
  return new FutureArray(defer<DeferredReduceOp>(op, a.get(), dim));
}

FACTORY_INIT(DeferredReduceOp, F_REDUCEOP);
} // namespace SHARPY
