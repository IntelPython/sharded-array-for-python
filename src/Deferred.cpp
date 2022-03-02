#include "include/ddptensor/Deferred.hpp"
#include <queue>

static std::queue<Deferred::ptr_type> _deferred;

Deferred::future_type Deferred::defer(Deferred::ptr_type && d)
{
    //auto f = d->get_future();
    _deferred.push(std::move(d));
    // return f;
    auto aa = Deferred::undefer_next();
    aa->run();
    return aa->get_future();
}

Deferred::ptr_type Deferred::undefer_next()
{
    auto r = std::move(_deferred.front());
    _deferred.pop();
    return r;
}
