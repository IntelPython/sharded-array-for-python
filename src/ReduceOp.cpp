#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"

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

struct DeferredReduceOp : public Deferred
 {
    tensor_i::future_type _a;
    dim_vec_type _dim;
    ReduceOpId _op;

    DeferredReduceOp(ReduceOpId op, const tensor_i::future_type & a, const dim_vec_type & dim)
        : _a(a), _dim(dim), _op(op)
    {}

    void run()
    {
        const auto a = std::move(_a.get());
        set_value(std::move(TypeDispatch<x::ReduceOp>(a, _op, _dim)));
    }
};

tensor_i::future_type ReduceOp::op(ReduceOpId op, const tensor_i::future_type & a, const dim_vec_type & dim)
{
    return defer<DeferredReduceOp>(op, a, dim);
}
