#include <mpi.h>
#include "ddptensor/ManipOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/CollComm.hpp"

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
    tensor_i::future_type _a;
    shape_type _shape;

    DeferredManipOp(tensor_i::future_type & a, const shape_type & shape)
        : _a(a), _shape(shape)
    {}

    void run()
    {
        auto a = std::move(_a.get());
        set_value(TypeDispatch<x::ManipOp>(a, _shape));
    }
};

tensor_i::future_type ManipOp::reshape(tensor_i::future_type & a, const shape_type & shape)
{
    return defer<DeferredManipOp>(a, shape);
}
