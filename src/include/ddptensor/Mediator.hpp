// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"

class NDSlice;

class Mediator
{
public:
    virtual ~Mediator() {}
    virtual uint64_t register_array(tensor_i::ptr_type ary) = 0;
    virtual void pull(rank_type from, const tensor_i::ptr_type & ary, const NDSlice & slice, void * buffer) = 0;
};

extern Mediator * theMediator;
