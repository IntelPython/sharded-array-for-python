#include "ddptensor/Creator.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Deferred.hpp"

namespace x {

    template<typename T>
    class Creator
    {
    public:
        using ptr_type = typename tensor_i::ptr_type;
        using typed_ptr_type = typename DPTensorX<T>::typed_ptr_type;

        static ptr_type op(CreatorId c, const shape_type & shp)
        {
            PVSlice pvslice(shp);
            shape_type shape(std::move(pvslice.shape_of_rank()));
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
                shape_type shape(std::move(pvslice.shape_of_rank()));
                auto a = xt::empty<T>(std::move(shape));
                a.fill(to_native<T>(v));
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
            }
            throw std::runtime_error("Unknown creator");
        }

        static ptr_type op(uint64_t start, uint64_t end, uint64_t step)
        {
            PVSlice pvslice({static_cast<uint64_t>(Slice(start, end, step).size())});
            auto lslc = pvslice.slice_of_rank();
            const auto & l1dslc = lslc.dim(0);
            auto a = xt::arange<T>(start + l1dslc._start*step, start + l1dslc._end * step, l1dslc._step);
            return operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
        }
    }; // class creatorx
} // namespace x

struct DeferredFromShape : public Deferred
{
    CreatorId _op;
    shape_type _shape;
    DTypeId _dtype;

    DeferredFromShape(CreatorId op, const shape_type & shape, DTypeId dtype)
        : _op(op), _shape(shape), _dtype(dtype)
    {}

    void run()
    {
        set_value(TypeDispatch<x::Creator>(_dtype, _op, _shape));
    }
};

tensor_i::future_type Creator::create_from_shape(CreatorId op, const shape_type & shape, DTypeId dtype)
{
    return defer<DeferredFromShape>(op, shape, dtype);
}

struct DeferredFull : public Deferred
{
    shape_type _shape;
    const py::object & _val;
    DTypeId _dtype;

    DeferredFull(const shape_type & shape, const py::object & val, DTypeId dtype)
        : _shape(shape), _val(val), _dtype(dtype)
    {}

    void run()
    {
        auto op = FULL;
        set_value(TypeDispatch<x::Creator>(_dtype, op, _shape, _val));
    }
};

tensor_i::future_type Creator::full(const shape_type & shape, const py::object & val, DTypeId dtype)
{
    return defer<DeferredFull>(shape, val, dtype);
}

struct DeferredArange : public Deferred
{
    uint64_t _start, _end, _step;
    DTypeId _dtype;

    DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype)
        : _start(start), _end(end), _step(step), _dtype(dtype)
    {}

    void run()
    {
        set_value(TypeDispatch<x::Creator>(_dtype, _start, _end, _step));
    };
};

tensor_i::future_type Creator::arange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype)
{
    return defer<DeferredArange>(start, end, step, dtype);
}
