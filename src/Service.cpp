#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Service.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/ddptensor.hpp"
#include "ddptensor/DDPTensorImpl.hpp"

#if 0
namespace x {
    struct Service
    {
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            if(a_ptr->is_replicated()) return a_ptr;
            if(a_ptr->has_owner() && a_ptr->slice().size() == 1) {
                if(getTransceiver()->rank() == a_ptr->owner()) {
                    a_ptr->_replica = *(xt::strided_view(a_ptr->xarray(), a_ptr->lslice()).begin());
                }
                getTransceiver()->bcast(&a_ptr->_replica, sizeof(T), a_ptr->owner());
                a_ptr->set_owner(REPLICATED);
            } else {
                throw(std::runtime_error("Replication implemented for single element and single owner only."));
            }
            return a_ptr;
        }
    };
}
#endif // if 0

struct DeferredService : public Deferred
{
    enum Op : int {
        REPLICATE,
        DROP,
        RUN,
        SERVICE_LAST
    };

    id_type _a;
    Op _op;

    DeferredService(Op op = SERVICE_LAST)
        : _a(), _op(op)
    {}
    DeferredService(Op op, const tensor_i::future_type & a)
        : _a(a.id()), _op(op)
    {}

    void run()
    {
        switch(_op) {
        case REPLICATE: {
            const auto a = std::move(Registry::get(_a).get());
            auto ddpt = dynamic_cast<DDPTensorImpl*>(a.get());
            assert(ddpt);
            ddpt->replicate();
            set_value(a);
            break;
        }
        case RUN:
            break;
        default:
           throw(std::runtime_error("Unkown Service operation requested."));
        }
    }

    bool generate_mlir(::mlir::OpBuilder & builder, ::mlir::Location loc, jit::DepManager & dm) override
    {
        switch(_op) {
        case DROP:
            dm.drop(_a);
            // FIXME create delete op and return it
            break;
        case RUN:
        case REPLICATE:
            return true;
        default:
            throw(std::runtime_error("Unkown Service operation requested."));
        }

        return false;
    }

    FactoryId factory() const
    {
        return F_SERVICE;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * Service::replicate(const ddptensor & a)
{
    return new ddptensor(defer<DeferredService>(DeferredService::REPLICATE, a.get()));
}

void Service::run()
{
    defer<DeferredService>(DeferredService::RUN);
    // defer_lambda([](){ return true; });
}

bool inited = false;
bool finied = false;

void Service::drop(const ddptensor & a)
{
    if(inited) {
        // if(getTransceiver()->is_spmd()) getTransceiver()->barrier();
        defer<DeferredService>(DeferredService::DROP, a.get());
    }
}

FACTORY_INIT(DeferredService, F_SERVICE);
