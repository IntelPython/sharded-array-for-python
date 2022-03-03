// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"

struct Deferred : tensor_i::promise_type
{
    using ptr_type = std::unique_ptr<Deferred>;
    using promise_type = tensor_i::promise_type;
    using future_type = tensor_i::future_type;

    Deferred() = default;
    Deferred(const Deferred &) = delete;

    virtual ~Deferred() {}
    virtual void run() = 0;

    static future_type defer(ptr_type &&);
    static ptr_type undefer_next();
};

template<typename T, typename... Ts>
Deferred::future_type defer(Ts&&... args)
{
    return Deferred::defer(std::make_unique<T>(args...));
}

struct UnDeferred : public Deferred
{
    UnDeferred(const tensor_i::ptr_type & ptr)
    {
        set_value(ptr);
    }

    void run()
    {
    }
};

extern void process_promises();
extern void sync();
