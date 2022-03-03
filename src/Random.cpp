#include <xtensor/xrandom.hpp>
#include "ddptensor/Random.hpp"
#include "ddptensor/x.hpp"

using ptr_type = tensor_i::ptr_type;

namespace x {

    template<typename T>
    struct Rand
    {
        template<typename L, typename U>
        static ptr_type op(const shape_type & shp, const L & lower, const U & upper)
        {
            if constexpr (std::is_floating_point<T>::value) {
                PVSlice pvslice(shp);
                shape_type shape(std::move(pvslice.shape_of_rank()));
                return operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::random::rand(std::move(shape), to_native<T>(lower), to_native<T>(upper))));
            }
        }
    };
}

struct DeferredRandomOp : public Deferred
{
    shape_type _shape;
    py::object _lower, _upper;
    DTypeId _dtype;

    DeferredRandomOp(DTypeId dtype, const shape_type & shape, const py::object & lower, const py::object & upper)
        : _shape(shape), _lower(lower), _upper(upper), _dtype(dtype)
    {}

    void run()
    {
        switch(_dtype) {
        case FLOAT64:
            set_value(x::Rand<double>::op(_shape, _lower, _upper));
            return;
        case FLOAT32:
            set_value(x::Rand<float>::op(_shape, _lower, _upper));
            return;
        }
        throw std::runtime_error("rand: dtype must be a floating point type");
    }
};

Random::future_type Random::rand(DTypeId dtype, const shape_type & shape, const py::object & lower, const py::object & upper)
{
    return defer<DeferredRandomOp>(dtype, shape, lower, upper);
}

void Random::seed(uint64_t s)
{
    xt::random::seed(s);
}
