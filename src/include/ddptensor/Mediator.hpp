// SPDX-License-Identifier: BSD-3-Clause

/*
  A high-level mediation between processes/ranks.
*/

#pragma once

#include "Deferred.hpp"
#include "tensor_i.hpp"
#include <vector>

class NDSlice;

class Mediator {
public:
  enum : uint64_t { LOCAL_ONLY = 0 };
  virtual ~Mediator() {}
  // virtual void pull(rank_type from, id_type guid, const NDSlice & slice, void
  // * buffer) = 0;
  virtual void to_workers(const Runable *dfrd) = 0;
};

extern void init_mediator(Mediator *);
extern void fini_mediator();
extern Mediator *getMediator();
