// SPDX-License-Identifier: BSD-3-Clause

/*
  Random number ops.
*/

#include "sharpy/Random.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include <bitsery/traits/vector.h>

namespace SHARPY {

using ptr_type = array_i::ptr_type;

#if 0
namespace x {

    template<typename T>
    struct Rand
    {
        //template<typename L, typename U>
        static ptr_type op(const shape_type & shp, T lower, T upper)
        {
            PVSlice pvslice(shp);
            shape_type shape(std::move(pvslice.tile_shape()));
            auto r = operatorx<T>::mk_tx(std::move(pvslice), std::move(xt::random::rand(std::move(shape), lower, upper)));
            return r;
        }
    };
}
#endif // if 0

struct DeferredRandomOp : public Deferred {
  shape_type _shape;
  double _lower, _upper;
  DTypeId _dtype;

  DeferredRandomOp() = default;
  DeferredRandomOp(const shape_type &shape, double lower, double upper,
                   DTypeId dtype)
      : _shape(shape), _lower(lower), _upper(upper), _dtype(dtype) {}

  void run() override {
#if 0
        switch(_dtype) {
        case FLOAT64:
            set_value(std::move(x::Rand<double>::op(_shape, _lower, _upper)));
            return;
        case FLOAT32:
            set_value(std::move(x::Rand<float>::op(_shape, static_cast<float>(_lower), static_cast<float>(_upper))));
            return;
        }
        throw std::runtime_error("rand: dtype must be a floating point type");
#endif // if 0
  }

  FactoryId factory() const override { return F_RANDOM; }

  template <typename S> void serialize(S &ser) {
    ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    ser.template value<sizeof(_lower)>(_lower);
    ser.template value<sizeof(_upper)>(_upper);
    ser.template value<sizeof(_dtype)>(_dtype);
  }
};

FutureArray *Random::rand(DTypeId dtype, const shape_type &shape,
                          const py::object &lower, const py::object &upper) {
  return new FutureArray(defer<DeferredRandomOp>(
      shape, to_native<double>(lower), to_native<double>(upper), dtype));
}

void Random::seed(uint64_t s) {
  // FIXME defer_lambda([s](){xt::random::seed(s); return
  // array_i::ptr_type();});
}

FACTORY_INIT(DeferredRandomOp, F_RANDOM);
} // namespace SHARPY
