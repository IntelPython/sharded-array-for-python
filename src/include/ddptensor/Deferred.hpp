// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"
#include "Registry.hpp"

struct Deferred : tensor_i::promise_type
{
    using ptr_type = std::unique_ptr<Deferred>;
    using promise_type = tensor_i::promise_type;
    using future_type = tensor_i::future_type;

    id_type _guid = Registry::NOGUID;

    Deferred() = default;
    Deferred(const Deferred &) = delete;
    future_type get_future();
    // void set_value(tensor_i::ptr_type &&);

    virtual ~Deferred() {}
    virtual void run() = 0;
    virtual FactoryId factory() const = 0;

    static future_type defer(ptr_type &&, bool);
    static ptr_type undefer_next();
    static void fini();
};

template<typename T, typename... Ts>
Deferred::future_type defer(Ts&&... args)
{
    return Deferred::defer(std::move(std::make_unique<T>(args...)), true);
}

struct UnDeferred : public Deferred
{
    UnDeferred(tensor_i::ptr_type ptr)
    {
        set_value(std::move(ptr));
    }

    void run()
    {
    }

    FactoryId factory() const
    {
        throw(std::runtime_error("No Factory for Undeferred."));
    }
};

template<typename L>
struct DeferredLambda : public Deferred
{
    L _l;

    DeferredLambda(L l)
        : _l(l)
    {}

    void run()
    {
        set_value(std::move(_l()));
    }

    FactoryId factory() const
    {
        throw(std::runtime_error("No Factory for DeferredLambda."));
    }
};

template<typename L>
Deferred::future_type defer(L && l)
{
    return Deferred::defer(std::move(std::make_unique<DeferredLambda<L>>(std::forward<L>(l))), false);
}

extern void process_promises();
extern void sync();
