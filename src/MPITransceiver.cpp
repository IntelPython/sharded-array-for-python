// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>
#include <limits>
#include "ddptensor/MPITransceiver.hpp"

MPITransceiver::MPITransceiver()
{
    int flag;
    MPI_Initialized(&flag);
    if(!flag) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error("Your MPI implementation is not MPI_THREAD_MULTIPLE. "
                                     "Please use a thread-safe MPI implementation.");
        }
    } else {
        MPI_Query_thread(&flag);
        if(flag != MPI_THREAD_MULTIPLE)
            throw(std::logic_error("MPI had been initialized incorrectly: not MPI_THREAD_MULTIPLE"));
        std::cerr << "MPI already initialized\n";
    }
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    _nranks = nranks;
    _rank = rank;
};

MPITransceiver::~MPITransceiver()
{
    int flag;
    MPI_Finalized(&flag);
    if(!flag)
        MPI_Finalize();
}

static MPI_Datatype to_mpi(DTypeId T)
{
    switch(T) {
    case FLOAT64: return MPI_DOUBLE;
    case FLOAT32: return MPI_FLOAT;
    case INT32:   return MPI_INT32_T;
    case INT64:   return MPI_INT64_T;
    case UINT32:  return MPI_INT32_T;
    case UINT64:  return MPI_INT64_T;
    case INT8:    return MPI_INT8_T;
    case UINT8:   return MPI_UINT8_T;
    case BOOL:    return MPI_C_BOOL;
    default: throw std::logic_error("unsupported data type");
    }
}

static MPI_Op to_mpi(RedOpType o)
{
    switch(o) {
    case MAX:  return MPI_MAX;
    case MIN:  return MPI_MIN;
    case SUM:  return MPI_SUM;
    case PROD: return MPI_PROD;
    // case OP_LAND: return MPI_LAND;
    // case OP_BAND: return MPI_BAND;
    // case OP_LOR:  return MPI_LOR;
    // case OP_BOR:  return MPI_BOR;
    // case OP_LXOR: return MPI_LXOR;
    // case OP_BXOR: return MPI_BXOR;
    default: throw std::logic_error("unsupported operation type");
    }
}



void MPITransceiver::barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPITransceiver::bcast(void * ptr, size_t N, rank_type root)
{
    MPI_Bcast(ptr, N, MPI_CHAR, root, MPI_COMM_WORLD);
}

void MPITransceiver::reduce_all(void * inout, DTypeId T, size_t N, RedOpType op)
{
    MPI_Allreduce(MPI_IN_PLACE, inout, N, to_mpi(T), to_mpi(op), MPI_COMM_WORLD);
}

void MPITransceiver::alltoall(const void* buffer_send,
                              const int* counts_send,
                              const int* displacements_send,
                              DTypeId datatype_send,
                              void* buffer_recv,
                              const int* counts_recv,
                              const int* displacements_recv,
                              DTypeId datatype_recv)
{
    MPI_Alltoallv(buffer_send,
                  counts_send,
                  displacements_send,
                  to_mpi(datatype_send),
                  buffer_recv,
                  counts_recv,
                  displacements_recv,
                  to_mpi(datatype_recv),
                  MPI_COMM_WORLD);
}

void MPITransceiver::send_recv(void* buffer_send,
                               int count_send,
                               DTypeId datatype_send,
                               int dest,
                               int source)
{
    constexpr int SRTAG = 505;
    MPI_Sendrecv_replace(buffer_send,
                         count_send,
                         to_mpi(datatype_send),
                         dest,
                         SRTAG,
                         source,
                         SRTAG,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
}
