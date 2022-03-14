#include <oneapi/tbb/concurrent_queue.h>
#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Transceiver.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"

static tbb::concurrent_bounded_queue<Deferred::ptr_type> _deferred;

Deferred::future_type Deferred::get_future()
{
    return {std::move(tensor_i::promise_type::get_future()), _guid};
}

#if 0
void Deferred::set_value(tensor_i::ptr_type && v)
{
    if(_guid != Registry::NOGUID) {
        Registry::put(_guid, v);
    }
    tensor_i::promise_type::set_value(std::forward<tensor_i::ptr_type>(v));
}
#endif

Deferred::future_type Deferred::defer(Deferred::ptr_type && d, bool is_global)
{
    if(is_global) {
        if(is_cw() && theTransceiver->rank() == 0) theMediator->to_workers(d);
        if(d) d->_guid = Registry::get_guid();
    }
    auto f = d ? d->get_future() : Deferred::future_type();
    Registry::put(f);
    _deferred.push(std::move(d));
    return f;
}

Deferred::ptr_type Deferred::undefer_next()
{
    Deferred::ptr_type r;
    _deferred.pop(r);
    return r;
}

void Deferred::fini()
{
    _deferred.clear();
}

void process_promises()
{
    while(true) {
        Deferred::ptr_type d;
        _deferred.pop(d);
        if(d) d->run();
        else break;
        d.reset();
    }
}

void sync()
{
    // FIXME this does not wait for the last deferred to complete
    while(!_deferred.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

