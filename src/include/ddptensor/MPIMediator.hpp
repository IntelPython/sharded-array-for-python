// SPDX-License-Identifier: BSD-3-Clause

/*
  A high-level mediation between processes/ranks implemented on top of MPI.
*/

#pragma once

#include "Mediator.hpp"
#include <mpi.h>
#include <thread>

class MPIMediator : public Mediator {
  std::thread *_listener;
  MPI_Comm _comm;

public:
  MPIMediator();
  virtual ~MPIMediator();
  // virtual void pull(rank_type from, id_type guid, const NDSlice & slice, void
  // * buffer);
  virtual void to_workers(const Runable *dfrd);

protected:
  void listen();
};
