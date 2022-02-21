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
    virtual void alltoall(const void* buffer_send,
                          const int* counts_send,
                          const int* displacements_send,
                          DTypeId datatype_send,
                          void* buffer_recv,
                          const int* counts_recv,
                          const int* displacements_recv,
                          DTypeId datatype_recv);

private:
    rank_type _nranks, _rank;
};
