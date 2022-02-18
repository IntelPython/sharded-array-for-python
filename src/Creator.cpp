#include "ddptensor/Operations.hpp"
#include "ddptensor/x.hpp"

namespace x {

    template<typename T>
    class Creator
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;
        using typed_ptr_type = typename DPTensorX<T>::typed_ptr_type;

        static ptr_type op(CreatorId c, const shape_type & shp)
        {
            PVSlice pvslice(shp);
            shape_type shape(std::move(pvslice.tile_shape()));
            switch(c) {
            case EMPTY:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::empty<T>(std::move(shape))));
            case ONES:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::ones<T>(std::move(shape))));
            case ZEROS:
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::zeros<T>(std::move(shape))));
            default:
                throw std::runtime_error("Unknown creator");
            };
        };

        template<typename V>
        static ptr_type op(CreatorId c, const shape_type & shp, const V & v)
        {
            if(c == FULL) {
                PVSlice pvslice(shp);
                shape_type shape(std::move(pvslice.tile_shape()));
                auto a = xt::empty<T>(std::move(shape));
                a.fill(to_native<T>(v));
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
            }
            throw std::runtime_error("Unknown creator");
        }
    }; // class creatorx
} // namespace x

tensor_i::ptr_type Creator::create_from_shape(CreatorId op, const shape_type & shape, DTypeId dtype)
{
    return TypeDispatch<x::Creator>(dtype, op, shape);
}

tensor_i::ptr_type Creator::full(const shape_type & shape, const py::object & val, DTypeId dtype)
{
    auto op = FULL;
    return TypeDispatch<x::Creator>(dtype, op, shape, val);
}
