// SPDX-License-Identifier: BSD-3-Clause

/*
  Communication device based on MPI.
*/

#pragma once

#include "Transceiver.hpp"
#include <mpi.h>

class MPITransceiver : public Transceiver {
public:
  MPITransceiver(bool is_cw);
  ~MPITransceiver();

  virtual bool is_cw() { return _is_cw && nranks() > 1; }

  virtual bool is_spmd() { return !_is_cw && nranks() > 1; }

  rank_type nranks() const { return _nranks; }

  rank_type rank() const { return _rank; }

  MPI_Comm comm() const { return _comm; }

  virtual void barrier();
  virtual void bcast(void *ptr, size_t N, rank_type root);
  virtual void reduce_all(void *inout, DTypeId T, size_t N, RedOpType op);
  virtual void alltoall(const void *buffer_send, const int *counts_send,
                        const int *displacements_send, DTypeId datatype_send,
                        void *buffer_recv, const int *counts_recv,
                        const int *displacements_recv);
  virtual void alltoall(const void *buffer_send, const int counts,
                        DTypeId datatype, void *buffer_recv);
  virtual void gather(void *buffer, const int *counts, const int *displacements,
                      DTypeId datatype, rank_type root);
  virtual void send_recv(void *buffer_send, int count_send,
                         DTypeId datatype_send, int dest, int source);

private:
  rank_type _nranks, _rank;
  MPI_Comm _comm;
  bool _is_cw;
};
