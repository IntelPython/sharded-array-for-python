// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thread>
#include "Mediator.hpp"

class MPIMediator : public Mediator
{
    std::thread _listener;

public:
    MPIMediator();
    virtual ~MPIMediator();
    virtual uint64_t register_array(tensor_i::ptr_type ary);
    virtual uint64_t unregister_array(uint64_t);
    virtual void pull(rank_type from, const tensor_i & ary, const NDSlice & slice, void * buffer);

protected:
    void listen();
};
