// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include "tensor_i.hpp"

class NDSlice;

class Mediator
{
public:
    enum : uint64_t {LOCAL_ONLY = 0};
    virtual ~Mediator() {}
    virtual uint64_t register_array(tensor_i::ptr_type ary) = 0;
    virtual uint64_t unregister_array(uint64_t) = 0;
    virtual void pull(rank_type from, const tensor_i & ary, const NDSlice & slice, void * buffer) = 0;
};

extern Mediator * theMediator;
