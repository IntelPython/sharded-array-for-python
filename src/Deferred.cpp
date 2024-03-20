// SPDX-License-Identifier: BSD-3-Clause

/*
  Creation/destruction of Deferreds.
  Implementation of worker loop processing deferred objects.
  This worker loop is executed in a separate thread until the system
  gets shut down.
*/

#include "include/sharpy/Deferred.hpp"
#include "include/sharpy/Mediator.hpp"
#include "include/sharpy/Registry.hpp"
#include "include/sharpy/Service.hpp"
#include "include/sharpy/Transceiver.hpp"
#include "include/sharpy/itac.hpp"
#include "include/sharpy/jit/mlir.hpp"

#include <oneapi/tbb/concurrent_queue.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

namespace SHARPY {

// thread-safe FIFO queue holding deferred objects
extern tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

// if needed, object/promise is broadcasted to worker processes
// (for controller/worker mode)
void _dist(const Runable *p) {
  if (getTransceiver() && getTransceiver()->is_cw() &&
      getTransceiver()->rank() == 0)
    getMediator()->to_workers(p);
}

// create a enriched future
Deferred::future_type Deferred::get_future() {
  return {promise_type::get_future().share(),
          _guid,
          _dtype,
          _shape,
          _device,
          _team};
}

// defer a array-producing computation by adding it to the queue.
// return a future for the resulting array.
// set is_global to false if result is a local temporary which does not need a
// guid
Deferred::future_type defer_array(Runable::ptr_type &&_d, bool is_global) {
  Deferred *d = dynamic_cast<Deferred *>(_d.get());
  if (!d)
    throw std::invalid_argument("Expected Deferred Array promise");
  if (is_global) {
    _dist(d);
    if (d->guid() == Registry::NOGUID) {
      d->set_guid(Registry::get_guid());
    }
  }
  auto f = d->get_future();
  Registry::put(f);
  push_runable(std::move(_d));
  return f;
}

// defer a global array producer
void Deferred::defer(Runable::ptr_type &&p) { defer_array(std::move(p), true); }

void Runable::defer(Runable::ptr_type &&p) { push_runable(std::move(p)); }

void Runable::fini() { _deferred.clear(); }

// process promises as they arrive through calls to defer
// This is run in a separate thread until shutdown is requested.
// Shutdown is indicated by a Deferred object which evaluates to false.
// The loop repeatedly creates MLIR functions for jit-compilation by letting
// Deferred objects add their MLIR code until an object can not produce MLIR
// but wants immediate execution (indicated by generate_mlir returning true).
// When execution is needed, the function signature (input args, return
// statement) is finalized, the function gets compiled and executed. The loop
// completes by calling run() on the requesting object.
void process_promises(const std::string &libidtr) {
  int vtProcessSym, vtSHARPYClass, vtPopSym;
  VT(VT_classdef, "sharpy", &vtSHARPYClass);
  VT(VT_funcdef, "process", vtSHARPYClass, &vtProcessSym);
  VT(VT_funcdef, "pop", vtSHARPYClass, &vtPopSym);
  VT(VT_begin, vtProcessSym);

  bool done = false;
  jit::JIT jit(libidtr);
  std::vector<Runable::ptr_type> deleters;

  do {
    // we need to keep runables/deferred/futures alive until we set their values
    // below
    std::vector<Runable::ptr_type> runables;

    jit::DepManager dm(jit);
    auto &builder = dm.getBuilder();
    auto loc = builder.getUnknownLoc();
    Runable::ptr_type d;

    if (!deleters.empty()) {
      for (auto &dl : deleters) {
        if (dl->generate_mlir(builder, loc, dm)) {
          assert(!"deleters must generate MLIR");
        }
        runables.emplace_back(std::move(dl));
      }
      deleters.clear();
    } else {
      while (true) {
        VT(VT_begin, vtPopSym);
        _deferred.pop(d);
        VT(VT_end, vtPopSym);
        if (d) {
          if (d->isDeleter()) {
            deleters.emplace_back(std::move(d));
          } else {
            if (d->generate_mlir(builder, loc, dm)) {
              break;
            };
            // keep alive for later set_value
            runables.emplace_back(std::move(d));
          }
        } else {
          // signals system shutdown
          done = true;
          break;
        }
      }
    }

    if (!runables.empty()) {
      dm.finalizeAndRun();
    } // no else needed

    // now we execute the deferred action which could not be compiled
    if (d) {
      py::gil_scoped_acquire acquire;
      d->run();
      d.reset();
    }
  } while (!done);
}
} // namespace SHARPY
