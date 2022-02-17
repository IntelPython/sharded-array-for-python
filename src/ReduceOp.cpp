#include "ddptensor/Operations.hpp"
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

tensor_i::ptr_type ReduceOp::op(ReduceOpId op, x::DPTensorBaseX::ptr_type a, const dim_vec_type & dim)
{
    return TypeDispatch<x::ReduceOp>(a, op, dim);
}
