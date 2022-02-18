// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Transceiver.hpp"

class MPITransceiver : public Transceiver
{
public:
    MPITransceiver();

    rank_type nranks() const
    {
        return _nranks;
    }

    rank_type rank() const
    {
        return _rank;
    }
    

    virtual void barrier();
    virtual void bcast(void * ptr, size_t N, rank_type root);
    virtual void reduce_all(void * inout, DTypeId T, size_t N, RedOpType op);

private:
    rank_type _nranks, _rank;
};
