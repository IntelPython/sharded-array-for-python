#include <mpi.h>
#include "ddptensor/ManipOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/CollComm.hpp"
#include "ddptensor/Factory.hpp"

namespace x {

    class ManipOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // Reshape
        // For now we always create a copy/new array.
        template<typename T>
        static ptr_type op(const shape_type & shape, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto b_ptr = x::operatorx<T>::mk_tx(shape);
            CollComm::coll_copy(b_ptr, a_ptr);
            return b_ptr;
        }
    };
}

struct DeferredManipOp : public Deferred
{
    id_type _a;
    shape_type _shape;

    DeferredManipOp() = default;
    DeferredManipOp(const tensor_i::future_type & a, const shape_type & shape)
        : _a(a.id()), _shape(shape)
    {}

    void run()
    {
        const auto a = std::move(Registry::get(_a));
        set_value(std::move(TypeDispatch<x::ManipOp>(a, _shape)));
    }

    FactoryId factory() const
    {
        return F_MANIPOP;
    }
    
    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template container<sizeof(shape_type::value_type)>(_shape, 8);
    }
};

tensor_i::future_type ManipOp::reshape(const tensor_i::future_type & a, const shape_type & shape)
{
    return defer<DeferredManipOp>(a, shape);
}

FACTORY_INIT(DeferredManipOp, F_MANIPOP);
