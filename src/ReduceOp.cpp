#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

    template<typename T>
    class ReduceOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"

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
            return operatorx<typename X::value_type>::mk_tx(new_shape, a, owner);
        }

        static ptr_type op(ReduceOpId rop, const ptr_type & a_ptr, const dim_vec_type & dims)
        {
            const auto _a = dynamic_cast<DPTensorX<T>*>(a_ptr.get());
            if(!_a )
                throw std::runtime_error("Invalid array object: could not dynamically cast");
            const auto & a = xt::strided_view(_a->xarray(), _a->lslice());

            switch(rop) {
            case MEAN:
                return dist_reduce(rop, _a->slice(), dims, xt::mean(a, dims));
            case PROD:
                return dist_reduce(rop, _a->slice(), dims, xt::prod(a, dims));
            case SUM:
                return dist_reduce(rop, _a->slice(), dims, xt::sum(a, dims));
            case STD:
                return dist_reduce(rop, _a->slice(), dims, xt::stddev(a, dims));
            case VAR:
                return dist_reduce(rop, _a->slice(), dims, xt::variance(a, dims));
            case MAX:
                return dist_reduce(rop, _a->slice(), dims, xt::amax(a, dims));
            case MIN:
                return dist_reduce(rop, _a->slice(), dims, xt::amin(a, dims));
            default:
                throw std::runtime_error("Unknown reduction operation");
            }
        }

#pragma GCC diagnostic pop

    };
} // namespace x

tensor_i::ptr_type ReduceOp::op(ReduceOpId op, x::DPTensorBaseX::ptr_type a, const dim_vec_type & dim)
{
    return TypeDispatch<x::ReduceOp>(a->dtype(), op, a, dim);
}
