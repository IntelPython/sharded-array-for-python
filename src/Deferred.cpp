// SPDX-License-Identifier: BSD-3-Clause

/*
  Creation/destruction of Deferreds.
  Implementation of worker loop processing deferred objects.
  This worker loop is executed in a separate thread until the system
  gets shut down.
*/

#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"
#include "include/ddptensor/Service.hpp"
#include "include/ddptensor/Transceiver.hpp"

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <oneapi/tbb/concurrent_queue.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

// thread-safe FIFO queue holding deferred objects
static tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

// add a deferred object to the queue
void push_runable(Runable::ptr_type &&r) { _deferred.push(std::move(r)); }

// if needed, object/promise is broadcasted to worker processes
// (for controller/worker mode)
void _dist(const Runable *p) {
  if (getTransceiver()->is_cw() && getTransceiver()->rank() == 0)
    getMediator()->to_workers(p);
}

// create a enriched future
Deferred::future_type Deferred::get_future() {
  return {std::move(promise_type::get_future().share()),
          _guid,
          _dtype,
          _shape,
          _team,
          _balanced};
}

// defer a tensor-producing computation by adding it to the queue.
// return a future for the resulting tensor.
// set is_global to false if result is a local temporary which does not need a
// guid
Deferred::future_type defer_tensor(Runable::ptr_type &&_d, bool is_global) {
  Deferred *d = dynamic_cast<Deferred *>(_d.get());
  if (!d)
    throw std::runtime_error("Expected Deferred Tensor promise");
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

// defer a global tensor producer
void Deferred::defer(Runable::ptr_type &&p) {
  defer_tensor(std::move(p), true);
}

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
void process_promises() {
  bool done = false;
  jit::JIT jit;
  do {
    ::mlir::OpBuilder builder(&jit.context());
    auto loc = builder.getUnknownLoc();

    // Create a MLIR module
    auto module = builder.create<::mlir::ModuleOp>(loc);
    auto protos = builder.create<::imex::dist::RuntimePrototypesOp>(loc);
    module.push_back(protos);
    // Create the jit func
    // create dummy type, we'll replace it with the actual type later
    auto dummyFuncType = builder.getFunctionType({}, {});
    if (false) {
      ::mlir::OpBuilder::InsertionGuard guard(builder);
      // Insert before module terminator.
      builder.setInsertionPoint(module.getBody(),
                                std::prev(module.getBody()->end()));
      auto func = builder.create<::mlir::func::FuncOp>(loc, "_debugFunc",
                                                       dummyFuncType);
      func.setPrivate();
    }
    std::string fname("ddpt_jit");
    auto function =
        builder.create<::mlir::func::FuncOp>(loc, fname, dummyFuncType);
    // create function entry block
    auto &entryBlock = *function.addEntryBlock();
    // Set the insertion point in the builder to the beginning of the function
    // body
    builder.setInsertionPointToStart(&entryBlock);
    // we need to keep runables/deferred/futures alive until we set their values
    // below
    std::vector<Runable::ptr_type> runables;

    jit::DepManager dm(function);

    Runable::ptr_type d;
    while (true) {
      _deferred.pop(d);
      if (d) {
        if (d->generate_mlir(builder, loc, dm)) {
          break;
        };
        // keep alive for later set_value
        runables.push_back(std::move(d));
      } else {
        // signals system shutdown
        done = true;
        break;
      }
    }

    if (!runables.empty()) {
      // get input buffers (before results!)
      auto input = std::move(dm.store_inputs());
      // create return statement and adjust function type
      uint64_t osz = dm.handleResult(builder);
      // also request generation of c-wrapper function
      function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                        builder.getUnitAttr());
      if (jit.verbose())
        function.getFunctionType().dump();
      // add the function to the module
      module.push_back(function);

      if (osz > 0 || !input.empty()) {
        // compile and run the module
        auto output = jit.run(module, fname, input, osz);
        if (output.size() != osz)
          throw std::runtime_error("failed running jit");

        // push results to deliver promises
        dm.deliver(output, osz);
      } else {
        if (jit.verbose())
          std::cerr << "\tskipping\n";
      }
    } // no else needed

    // now we execute the deferred action which could not be compiled
    if (d) {
      py::gil_scoped_acquire acquire;
      d->run();
      d.reset();
    }
  } while (!done);
}
