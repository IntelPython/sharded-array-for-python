#include <xtensor/xrandom.hpp>
#include "ddptensor/Random.hpp"
#include "ddptensor/x.hpp"

using ptr_type = tensor_i::ptr_type;

namespace x {

    template<typename T>
    struct Rand
    {
        //template<typename L, typename U>
        static ptr_type op(const shape_type & shp, T lower, T upper)
        {
            PVSlice pvslice(shp);
            shape_type shape(std::move(pvslice.shape_of_rank()));
            return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::random::rand(std::move(shape), lower, upper)));
        }
    };
}

template<typename T>
struct DeferredRandomOp : public Deferred
{
    shape_type _shape;
    T _lower, _upper;

    DeferredRandomOp(const shape_type & shape, T lower, T upper)
        : _shape(shape), _lower(lower), _upper(upper)
    {}

    void run()
    {
        set_value(x::Rand<T>::op(_shape, _lower, _upper));
    }
};

Random::future_type Random::rand(DTypeId dtype, const shape_type & shape, const py::object & lower, const py::object & upper)
{
    switch(dtype) {
    case FLOAT64: {
        double lo = x::to_native<double>(lower);
        double up = x::to_native<double>(upper);
        return  defer([shape, lo, up](){return x::Rand<double>::op(shape, lo, up);});
        //return defer<DeferredRandomOp<double>>(shape, x::to_native<double>(lower), x::to_native<double>(upper));
    }
    case FLOAT32: {
        float lo = x::to_native<float>(lower);
        float up = x::to_native<float>(upper);
        return  defer([shape, lo, up](){return x::Rand<float>::op(shape, lo, up);});
        //return defer<DeferredRandomOp<float>>(shape, x::to_native<double>(lower), x::to_native<double>(upper));
    }
    default:
        throw std::runtime_error("rand: dtype must be a floating point type");
    }
}

void Random::seed(uint64_t s)
{
    defer([s](){xt::random::seed(s); return tensor_i::ptr_type();});
}
