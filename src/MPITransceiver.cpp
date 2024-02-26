// SPDX-License-Identifier: BSD-3-Clause

/*
  Communication device based on MPI.
*/

#include "sharpy/MPITransceiver.hpp"
#include <iostream>
#include <limits>
#include <mpi.h>
#include <sstream>

namespace SHARPY {

// Init MPI and transceiver

MPITransceiver::MPITransceiver(bool is_cw)
    : _nranks(1), _rank(0), _comm(MPI_COMM_WORLD), _is_cw(is_cw) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      throw std::runtime_error(
          "Your MPI implementation is not MPI_THREAD_MULTIPLE. "
          "Please use a thread-safe MPI implementation.");
    }
  } else {
    MPI_Query_thread(&flag);
    if (flag != MPI_THREAD_MULTIPLE)
      throw(std::logic_error(
          "MPI had been initialized incorrectly: not MPI_THREAD_MULTIPLE"));
    std::cerr << "MPI already initialized\n";
  }

  int nranks, rank;
  MPI_Comm_rank(_comm, &rank);
  MPI_Comm parentComm;
  MPI_Comm_get_parent(&parentComm);

  // rank father-of-all checks if he's requested to spawn processes:
  if (rank == 0 && parentComm == MPI_COMM_NULL) {
    // Ok, let's spawn the clients.
    // I need some information for the startup.
    // 1. Name of the executable (default is the current exe)
    const char *_tmp = getenv("SHARPY_MPI_SPAWN");
    if (_tmp) {
      int nClientsToSpawn = atol(_tmp);
      std::string clientExe;
      std::vector<std::string> args;
      _tmp = getenv("SHARPY_MPI_EXECUTABLE");
      if (!_tmp) {
        _tmp = getenv("PYTHON_EXE");
        if (!_tmp)
          throw std::runtime_error("Spawning MPI processes requires setting "
                                   "'SHARPY_MPI_EXECUTABLE' or 'PYTHON_EXE'");
        clientExe = _tmp;
        // 2. arguments
        _tmp = "-c import sharpy as sp; sp.init(True)";
        args.push_back("-c");
        args.push_back("import sharpy as sp; sp.init(True)");
      } else {
        clientExe = _tmp;
        // 2. arguments
        _tmp = getenv("SHARPY_MPI_EXE_ARGS");
        if (_tmp) {
          std::istringstream iss(_tmp);
          std::copy(std::istream_iterator<std::string>(iss),
                    std::istream_iterator<std::string>(),
                    std::back_inserter(args));
        }
      }

      const char *clientArgs[args.size() + 1];
      for (auto i = 0ul; i < args.size(); ++i)
        clientArgs[i] = args[i].c_str();
      clientArgs[args.size()] = nullptr;

      // 3. Special setting for MPI_Info: hosts
      const char *clientHost = getenv("SHARPY_MPI_HOSTS");

      // Prepare MPI_Info object:
      MPI_Info clientInfo = MPI_INFO_NULL;
      if (clientHost) {
        MPI_Info_create(&clientInfo);
        MPI_Info_set(clientInfo, const_cast<char *>("host"),
                     const_cast<char *>(clientHost));
        std::cerr << "[SHARPY " << rank << "] Set MPI_Info_set(\"host\", \""
                  << clientHost << "\")\n";
      }
      // Now spawn the client processes:
      std::cerr << "[SHARPY " << rank << "] Spawning " << nClientsToSpawn
                << " MPI processes (" << clientExe << " " << _tmp << ")"
                << std::endl;
      int *errCodes = new int[nClientsToSpawn];
      MPI_Comm interComm;
      int err =
          MPI_Comm_spawn(const_cast<char *>(clientExe.c_str()),
                         const_cast<char **>(clientArgs), nClientsToSpawn,
                         clientInfo, 0, MPI_COMM_WORLD, &interComm, errCodes);
      delete[] errCodes;
      if (err) {
        std::cerr << "[SHARPY " << rank
                  << "] Error in MPI_Comm_spawn. Skipping process spawning";
      } else {
        MPI_Intercomm_merge(interComm, 0, &_comm);
      }
    } // else {
      // No process spawning
      // MPI-1 situation: all clients to be started by mpiexec
      //                    _comm = MPI_COMM_WORLD;
      //}
  }
  if (parentComm != MPI_COMM_NULL) {
    // I am a child. Build intra-comm to the parent.
    MPI_Intercomm_merge(parentComm, 1, &_comm);
  }

  MPI_Comm_size(_comm, &nranks);
  MPI_Comm_rank(_comm, &rank);
  _nranks = nranks;
  _rank = rank;
};

MPITransceiver::~MPITransceiver() {
  int flag;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
}

// convert sharpy's dtype to MPI datatype
static MPI_Datatype to_mpi(DTypeId T) {
  switch (T) {
  case FLOAT64:
    return MPI_DOUBLE;
  case FLOAT32:
    return MPI_FLOAT;
  case INT32:
    return MPI_INT32_T;
  case INT64:
    return MPI_INT64_T;
  case UINT32:
    return MPI_INT32_T;
  case UINT64:
    return MPI_INT64_T;
  case INT8:
    return MPI_INT8_T;
  case UINT8:
    return MPI_UINT8_T;
  case BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("unsupported data type");
  }
}

// convert sharpy's RedOpType into MPI_op
static MPI_Op to_mpi(RedOpType o) {
  switch (o) {
  case MAX:
    return MPI_MAX;
  case MIN:
    return MPI_MIN;
  case SUM:
    return MPI_SUM;
  case PROD:
    return MPI_PROD;
  // case OP_LAND: return MPI_LAND;
  // case OP_BAND: return MPI_BAND;
  // case OP_LOR:  return MPI_LOR;
  // case OP_BOR:  return MPI_BOR;
  // case OP_LXOR: return MPI_LXOR;
  // case OP_BXOR: return MPI_BXOR;
  default:
    throw std::logic_error("unsupported operation type");
  }
}

void MPITransceiver::barrier() { MPI_Barrier(_comm); }

void MPITransceiver::bcast(void *ptr, size_t N, rank_type root) {
  MPI_Bcast(ptr, N, MPI_CHAR, root, _comm);
}

void MPITransceiver::reduce_all(void *inout, DTypeId T, size_t N,
                                RedOpType op) {
  MPI_Allreduce(MPI_IN_PLACE, inout, N, to_mpi(T), to_mpi(op), _comm);
}

Transceiver::WaitHandle
MPITransceiver::alltoall(const void *buffer_send, const int *counts_send,
                         const int *displacements_send, DTypeId datatype,
                         void *buffer_recv, const int *counts_recv,
                         const int *displacements_recv) {
  MPI_Request request;
  MPI_Ialltoallv(buffer_send, counts_send, displacements_send, to_mpi(datatype),
                 buffer_recv, counts_recv, displacements_recv, to_mpi(datatype),
                 _comm, &request);
  static_assert(sizeof(request) == sizeof(WaitHandle));
  return static_cast<WaitHandle>(request);
}

void MPITransceiver::alltoall(const void *buffer_send, const int counts,
                              DTypeId datatype, void *buffer_recv) {
  MPI_Alltoall(buffer_send, counts, to_mpi(datatype), buffer_recv, counts,
               to_mpi(datatype), _comm);
}

void MPITransceiver::gather(void *buffer, const int *counts,
                            const int *displacements, DTypeId datatype,
                            rank_type root) {
  auto dtype = to_mpi(datatype);
  if (root == REPLICATED) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, dtype, buffer, counts, displacements, dtype,
                   _comm);
  } else {
    if (root == _rank) {
      MPI_Gatherv(MPI_IN_PLACE, 0, dtype, buffer, counts, displacements, dtype,
                  root, _comm);
    } else {
      MPI_Gatherv(buffer, counts[_rank], dtype, nullptr, nullptr, nullptr,
                  dtype, root, _comm);
    }
  }
}

void MPITransceiver::send_recv(void *buffer_send, int count_send,
                               DTypeId datatype_send, int dest, int source) {
  constexpr int SRTAG = 505;
  MPI_Sendrecv_replace(buffer_send, count_send, to_mpi(datatype_send), dest,
                       SRTAG, source, SRTAG, _comm, MPI_STATUS_IGNORE);
}

void MPITransceiver::wait(WaitHandle h) {
  if (h) {
    auto r = static_cast<MPI_Request>(h);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
  }
}
} // namespace SHARPY
