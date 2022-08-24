// SPDX-License-Identifier: BSD-3-Clause

// Implementation of reduction operations

#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
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
                theTransceiver->reduce_all(a.data(), DTYPE<typename X::value_type>::value, len, rop);
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
static ::imex::ptensor::ReduceOpId ddpt2mlir(const ReduceOpId rop)
{
    switch(rop) {
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

struct DeferredReduceOp : public Deferred
{
    id_type _a;
    dim_vec_type _dim;
    ReduceOpId _op;

    DeferredReduceOp() = default;
    DeferredReduceOp(ReduceOpId op, const tensor_i::future_type & a, const dim_vec_type & dim)
        : Deferred(a.dtype(), 0), // FIXME rank
          _a(a.id()), _dim(dim), _op(op)
    {}

    void run()
    {
#if 0 
        const auto a = std::move(Registry::get(_a).get());
        set_value(std::move(TypeDispatch<x::ReduceOp>(a, _op, _dim)));
#endif
    }

    ::mlir::Value generate_mlir(::mlir::OpBuilder & builder, ::mlir::Location loc, jit::IdValueMap & ivm) override
    {
        // FIXME reduction over individual dimensions is not supported
        auto a = ivm[_a].first;
        auto a_ptt = a.getType().dyn_cast<::imex::ptensor::PTensorType>();
        assert(a_ptt);
        
        auto rtyp = ::imex::ptensor::PTensorType::get(
            builder.getContext(), 
            ::mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(), a_ptt.getRtensor().getElementType()),
            true
        );
        auto rop = builder.create<::imex::ptensor::ReductionOp>(loc, rtyp, builder.getI32IntegerAttr(ddpt2mlir(_op)), a);
        auto setter = [this](uint64_t rank, void *allocated, void *aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides) {
            this->set_value(std::move(mk_tnsr(_dtype, rank, allocated, aligned, offset, sizes, strides)));
        };
        ivm[_guid] = {rop, setter};
        return rop;
    }

    FactoryId factory() const
    {
        return F_REDUCEOP;
    }
    
    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template container<sizeof(dim_vec_type::value_type)>(_dim, 8);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * ReduceOp::op(ReduceOpId op, const ddptensor & a, const dim_vec_type & dim)
{
    return new ddptensor(defer<DeferredReduceOp>(op, a.get(), dim));
}

FACTORY_INIT(DeferredReduceOp, F_REDUCEOP);
