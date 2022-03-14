#include "ddptensor/Creator.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Factory.hpp"

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

        static ptr_type op(CreatorId c, const shape_type & shp, PyScalar v)
        {
            T val;
            if constexpr (std::is_integral<T>::value) val = static_cast<T>(v._int);
            else if constexpr (std::is_floating_point<T>::value) val = static_cast<T>(v._float);
            if(c == FULL) {
                if(VPROD(shp) <= 1) {
                    return operatorx<T>::mk_tx(val, REPLICATED);
                }
                PVSlice pvslice(shp);
                shape_type shape(std::move(pvslice.tile_shape()));
                auto a = xt::empty<T>(std::move(shape));
                a.fill(val);
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
            auto r = operatorx<T>::mk_tx(std::move(pvslice), std::move(a));
            return r;
        }
    }; // class creatorx
} // namespace x

struct DeferredFromShape : public Deferred
{
    shape_type _shape;
    DTypeId _dtype;
    CreatorId _op;

    DeferredFromShape() = default;
    DeferredFromShape(CreatorId op, const shape_type & shape, DTypeId dtype)
        : _shape(shape), _dtype(dtype), _op(op)
    {}

    void run()
    {
        set_value(std::move(TypeDispatch<x::Creator>(_dtype, _op, _shape)));
    }

    FactoryId factory() const
    {
        return F_FROMSHAPE;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
        ser.template value<sizeof(_dtype)>(_dtype);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * Creator::create_from_shape(CreatorId op, const shape_type & shape, DTypeId dtype)
{
    return new ddptensor(defer<DeferredFromShape>(op, shape, dtype));
}

struct DeferredFull : public Deferred
{
    shape_type _shape;
    PyScalar _val;
    DTypeId _dtype;

    DeferredFull() = default;
    DeferredFull(const shape_type & shape, PyScalar val, DTypeId dtype)
        : _shape(shape), _val(val), _dtype(dtype)
    {}

    void run()
    {
        auto op = FULL;
        set_value(std::move(TypeDispatch<x::Creator>(_dtype, op, _shape, _val)));
    }

    FactoryId factory() const
    {
        return F_FULL;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
        ser.template value<sizeof(_val)>(_val._int);
        ser.template value<sizeof(_dtype)>(_dtype);
    }
};

ddptensor * Creator::full(const shape_type & shape, const py::object & val, DTypeId dtype)
{
    auto v = mk_scalar(val, dtype);
    return new ddptensor(defer<DeferredFull>(shape, v, dtype));
}

struct DeferredArange : public Deferred
{
    uint64_t _start, _end, _step;
    DTypeId _dtype;

    DeferredArange() = default;
    DeferredArange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype)
        : _start(start), _end(end), _step(step), _dtype(dtype)
    {}

    void run()
    {
        set_value(std::move(TypeDispatch<x::Creator>(_dtype, _start, _end, _step)));
    };

    FactoryId factory() const
    {
        return F_ARANGE;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_start)>(_start);
        ser.template value<sizeof(_end)>(_end);
        ser.template value<sizeof(_step)>(_step);        
        ser.template value<sizeof(_dtype)>(_dtype);
    }
};

ddptensor * Creator::arange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype)
{
    return new ddptensor(defer<DeferredArange>(start, end, step, dtype));
}

ddptensor * Creator::mk_future(const py::object & b)
{
    if(py::isinstance<ddptensor>(b)) {
        return b.cast<ddptensor*>();
    } else if(py::isinstance<py::float_>(b)) {
        return Creator::full({1}, b, FLOAT64);
    } else if(py::isinstance<py::int_>(b)) {
        return Creator::full({1}, b, INT64);
    }
    throw std::runtime_error("Invalid right operand to elementwise binary operation");
};

FACTORY_INIT(DeferredFromShape, F_FROMSHAPE);
FACTORY_INIT(DeferredFull, F_FULL);
FACTORY_INIT(DeferredArange, F_ARANGE);
