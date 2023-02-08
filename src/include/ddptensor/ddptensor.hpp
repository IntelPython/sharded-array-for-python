// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"
#include "Service.hpp"

class ddptensor
{
    tensor_i::future_type _ftx;

public:
    ddptensor(const tensor_i::future_type & f)
        : _ftx(f)
    {}
    ddptensor(std::shared_future<tensor_i::ptr_type> && f, id_type id, DTypeId dt, int rank, bool balanced)
        : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt, rank, balanced)
    {}

    ~ddptensor()
    {
        Service::drop(*this);
    }

    const tensor_i::future_type & get() const
    {
        return _ftx;
    }
};
