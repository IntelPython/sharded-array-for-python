// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thread>
#include "Mediator.hpp"

class MPIMediator : public Mediator
{
    std::thread * _listener;

public:
    MPIMediator();
    virtual ~MPIMediator();
    virtual void pull(rank_type from, id_type guid, const NDSlice & slice, void * buffer);
    virtual void to_workers(const Deferred::ptr_type & dfrd);

protected:
    void listen();
};
