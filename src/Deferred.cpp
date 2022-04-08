// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/tbb/concurrent_queue.h>
#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Transceiver.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"

static tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

void push_runable(Runable::ptr_type && r)
{
    _deferred.push(std::move(r));
}

void _dist(const Runable * p)
{
    if(is_cw() && theTransceiver->rank() == 0)
        theMediator->to_workers(p);
}

Deferred::future_type Deferred::get_future()
{
    return {std::move(tensor_i::promise_type::get_future().share()), _guid};
}

Deferred::future_type defer_tensor(Runable::ptr_type && _d, bool is_global)
{
    Deferred * d = dynamic_cast<Deferred*>(_d.get());
    if(!d) throw std::runtime_error("Expected Deferred Tensor promise");
    if(is_global) {
        _dist(d);
        d->_guid = Registry::get_guid();
    }
    auto f = d->get_future();
    Registry::put(f);
    push_runable(std::move(_d));
    return f;
}

void Deferred::defer(Runable::ptr_type && p)
{
    defer_tensor(std::move(p), true);
}

void Runable::defer(Runable::ptr_type && p)
{
    push_runable(std::move(p));
}

void Runable::fini()
{
    _deferred.clear();
}

void process_promises()
{
    while(true) {
        Runable::ptr_type d;
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

