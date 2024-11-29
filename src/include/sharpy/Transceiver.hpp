// SPDX-License-Identifier: BSD-3-Clause

/*
  Abstract communication device.
*/

#pragma once

#include "CppTypes.hpp"

namespace SHARPY {

class Transceiver {
public:
  using WaitHandle = uint32_t;

  virtual ~Transceiver() {};

  virtual bool is_cw() = 0;
  virtual bool is_spmd() = 0;

  virtual rank_type nranks() const = 0;
  virtual rank_type rank() const = 0;

  virtual std::string mesh() const = 0;

  // Barrier
  virtual void barrier() = 0;

  // Broadcast data from root to all other processes
  // @param[inout] ptr   on root: pointer to data to be sent
  //                     on all other processes: pointer to buffer to store
  //                     received data
  // @param[in]    N     number of bytes in ptr
  // @param[in]    root  process id which collects data
  virtual void bcast(void *ptr, size_t N, rank_type root) = 0;

  // Element-wise reduce given array with given operation and provide result on
  // all processes
  // @param[inout] inout input to reduction and result
  // @param[in]    T     data type of elements in inout
  // @param[in]    N     number of elements in inout
  // @param[in]    op    reduction operation
  virtual void reduce_all(void *inout, DTypeId T, size_t N, RedOpType op) = 0;

  // umm, can this be higher-level?
  virtual WaitHandle alltoall(const void *buffer_send, const int *counts_send,
                              const int *displacements_send,
                              DTypeId datatype_send, void *buffer_recv,
                              const int *counts_recv,
                              const int *displacements_recv) = 0;
  virtual void alltoall(const void *buffer_send, const int counts,
                        DTypeId datatype, void *buffer_recv) = 0;

  virtual void gather(void *buffer, const int *counts, const int *displacements,
                      DTypeId datatype, rank_type root) = 0;

  virtual void send_recv(void *buffer_send, int count_send,
                         DTypeId datatype_send, int dest, int source) = 0;
  virtual void wait(WaitHandle) = 0;
};

extern void init_transceiver(Transceiver *);
extern void fini_transceiver();
extern Transceiver *getTransceiver();
} // namespace SHARPY
