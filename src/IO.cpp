#include "ddptensor/IO.hpp"
#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Transceiver.hpp"

using promise_type = std::promise<py::object>;
using future_type = std::shared_future<py::object>;

struct DeferredToNumpy : public DeferredT<promise_type, future_type>
{
    id_type _a;

    DeferredToNumpy() = default;
    DeferredToNumpy(const tensor_i::future_type & a)
        : _a(a.id())
    {}

    void run()
    {
        const auto a = std::move(Registry::get(_a).get());
        set_value(GetItem::do_gather(a, is_cw() ? 0 : REPLICATED));
    }

    FactoryId factory() const
    {
        return F_TONUMPY;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
    }
};

py::object IO::to_numpy(const ddptensor & a)
{
    assert(!is_cw() || theTransceiver->rank() == 0);
    auto f = defer<DeferredToNumpy>(a.get());
    auto x = f.get();
    return x;
}

FACTORY_INIT(DeferredToNumpy, F_TONUMPY);
