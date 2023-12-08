// SPDX-License-Identifier: BSD-3-Clause

/*
  Initing and Finalizing mediator singleton.
*/

#include <sharpy/Mediator.hpp>

namespace SHARPY {

static Mediator *theMediator = nullptr;

void init_mediator(Mediator *t) {
  if (theMediator)
    delete theMediator;
  theMediator = t;
}

void fini_mediator() {
  if (theMediator)
    delete theMediator;
  theMediator = nullptr;
}

Mediator *getMediator() { return theMediator; }
} // namespace SHARPY
