#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Replicate.hpp"
#include "ddptensor/Factory.hpp"

namespace x {
    struct Replicate
    {
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            if(a_ptr->is_replicated()) return a_ptr;
            if(a_ptr->has_owner() && a_ptr->slice().size() == 1) {
                if(theTransceiver->rank() == a_ptr->owner()) {
                    a_ptr->_replica = *(xt::strided_view(a_ptr->xarray(), a_ptr->lslice()).begin());
                }
                theTransceiver->bcast(&a_ptr->_replica, sizeof(T), a_ptr->owner());
                a_ptr->set_owner(REPLICATED);
            } else {
                throw(std::runtime_error("Replication implemented for single element and single owner only."));
            }
            return a_ptr;
        }
    };
}

struct DeferredReplicate : public Deferred
{
    id_type _a;

    DeferredReplicate() = default;
    DeferredReplicate(const tensor_i::future_type & a)
        : _a(a.id())
    {}

    void run()
    {
        const auto a = std::move(Registry::get(_a));
        set_value(std::move(TypeDispatch<x::Replicate>(a)));
    }

    FactoryId factory() const
    {
        return F_REPLICATE;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
    }
};

tensor_i::future_type Replicate::replicate(const tensor_i::future_type & a)
{
    return defer<DeferredReplicate>(a);
}

FACTORY_INIT(DeferredReplicate, F_REPLICATE);
