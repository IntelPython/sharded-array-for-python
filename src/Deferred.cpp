#include "include/ddptensor/Deferred.hpp"
#include <oneapi/tbb/concurrent_queue.h>

static tbb::concurrent_bounded_queue<Deferred::ptr_type> _deferred;

Deferred::future_type Deferred::defer(Deferred::ptr_type && d)
{
    auto f = d ? d->get_future() : Deferred::future_type();
    _deferred.push(std::move(d));
    return f;
    /* auto aa = Deferred::undefer_next();
    aa->run();
    return aa->get_future(); */
}

Deferred::ptr_type Deferred::undefer_next()
{
    Deferred::ptr_type r;
    _deferred.pop(r);
    return r;
}

void process_promises()
{
    while(true) {
        Deferred::ptr_type d;
        _deferred.pop(d);
        // auto d = std::move(Deferred::undefer_next());
        if(d) d->run();
        else break;
        d.reset();
    }
}

void sync()
{
    while(!_deferred.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
