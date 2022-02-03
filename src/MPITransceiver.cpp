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
    }
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    _nranks = nranks;
    _rank = rank;
};

static MPI_Datatype to_mpi(DType T)
{
    switch(T) {
    case DT_FLOAT64: return MPI_DOUBLE;
    case DT_FLOAT32: return MPI_FLOAT;
    case DT_INT32:   return MPI_INT32_T;
    case DT_INT64:   return MPI_INT64_T;
    case DT_UINT32:  return MPI_INT32_T;
    case DT_UINT64:  return MPI_INT64_T;
    case DT_INT8:    return MPI_INT8_T;
    case DT_UINT8:   return MPI_UINT8_T;
    case DT_BOOL:    return MPI_C_BOOL;
    default: throw std::logic_error("unsupported data type");
    }
}

static MPI_Op to_mpi(RedOpType o)
{
    switch(o) {
    case OP_MAX:  return MPI_MAX;
    case OP_MIN:  return MPI_MIN;
    case OP_SUM:  return MPI_SUM;
    case OP_PROD: return MPI_PROD;
    case OP_LAND: return MPI_LAND;
    case OP_BAND: return MPI_BAND;
    case OP_LOR:  return MPI_LOR;
    case OP_BOR:  return MPI_BOR;
    case OP_LXOR: return MPI_LXOR;
    case OP_BXOR: return MPI_BXOR;
    default: throw std::logic_error("unsupported operation type");
    }
}

void MPITransceiver::bcast(void * ptr, size_t N, rank_type root)
{
    MPI_Bcast(ptr, N, MPI_CHAR, root, MPI_COMM_WORLD);
}

void MPITransceiver::reduce_all(void * inout, DType T, size_t N, RedOpType op)
{
    MPI_Allreduce(MPI_IN_PLACE, inout, N, to_mpi(T), to_mpi(op), MPI_COMM_WORLD);
}
