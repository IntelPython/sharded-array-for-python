// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include "tensor_i.hpp"
#include "Deferred.hpp"

class NDSlice;

class Mediator
{
public:
    enum : uint64_t {LOCAL_ONLY = 0};
    virtual ~Mediator() {}
    virtual void pull(rank_type from, id_type guid, const NDSlice & slice, void * buffer) = 0;
    virtual void to_workers(const Deferred::ptr_type & dfrd) = 0;
};

extern Mediator * theMediator;
