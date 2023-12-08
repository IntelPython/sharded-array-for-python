// SPDX-License-Identifier: BSD-3-Clause

/*
  Operations on arrays may be deferred so that several of them can be
  jit-compiled together. Each operation is represented as an object of type
  "Deferred". A deferred object is a promise and a "Runnable". The promise gives
  access to a future so that users can wait for the promise to provide the
  value. Runnable is the interface allowing promises to execute and/or generate
  MLIR.
*/

#pragma once

#include "Registry.hpp"
#include "Transceiver.hpp"
#include "array_i.hpp"

namespace mlir {
class OpBuilder;
class Location;
} // namespace mlir

namespace SHARPY {

namespace jit {
class DepManager;
}

extern void process_promises();

// interface for promises/tasks to generate MLIR or execute immediately.
struct Runable {
  using ptr_type = std::unique_ptr<Runable>;
  virtual ~Runable(){};
  /// actually execute, a deferred will set value of future
  virtual void run() {
    throw(std::runtime_error(
        "No immediate execution support for this operation."));
  };
  /// generate MLIR code for jit
  /// the runable might not generate MLIR and instead return true
  /// to request the scheduler to execute the run method instead.
  /// @return false on success and true to request execution of run()
  virtual bool generate_mlir(::mlir::OpBuilder &, const ::mlir::Location &,
                             jit::DepManager &) {
    throw(std::runtime_error("No MLIR support for this operation."));
    return false;
  };
  virtual FactoryId factory() const = 0;
  virtual void defer(ptr_type &&);
  static void fini();
};

extern void push_runable(Runable::ptr_type &&r);

// helper class
// FIXME team is currently set to getTransceiver() always
template <typename P, typename F> struct DeferredT : public P, public Runable {
  using ptr_type = std::unique_ptr<DeferredT>;
  using promise_type = P;
  using future_type = F;

  DeferredT() = default;
  DeferredT(const DeferredT<P, F> &) = delete;
  DeferredT(DTypeId dt, shape_type &&shape, std::string &&device, uint64_t team,
            id_type guid = Registry::NOGUID)
      : P(guid, dt, std::forward<shape_type>(shape),
          std::forward<std::string>(device),
          team ? reinterpret_cast<uint64_t>(getTransceiver()) : 0),
        Runable() {}
  DeferredT(DTypeId dt, shape_type &&shape, const std::string &device,
            uint64_t team, id_type guid = Registry::NOGUID)
      : P(guid, dt, std::forward<shape_type>(shape), device,
          team ? reinterpret_cast<uint64_t>(getTransceiver()) : 0),
        Runable() {}
  DeferredT(DTypeId dt, const shape_type &shape = {},
            const std::string &device = {}, uint64_t team = 0,
            id_type guid = Registry::NOGUID)
      : P(guid, dt, shape, device,
          team ? reinterpret_cast<uint64_t>(getTransceiver()) : 0),
        Runable() {}
};

/// Deferred operation returning/producing a array
/// holds a guid as well as shape, dtype, device and team of future array
class Deferred
    : public DeferredT<array_i::promise_type, array_i::future_type> {
public:
  using ptr_type = std::unique_ptr<Deferred>;
  using DeferredT<array_i::promise_type, array_i::future_type>::DeferredT;

  // FIXME we should not allow default values for dtype and shape
  // we should need this only while we are gradually moving to mlir
  Deferred(const Deferred &) = delete;
  Deferred(Deferred &&) = delete;

  future_type get_future();
  // from Runable
  void defer(Runable::ptr_type &&);
};

extern void _dist(const Runable *p);

// defer operations which do *not* return a array, e.g. which are not a
// Deferred
template <typename T, typename... Ts,
          std::enable_if_t<!std::is_base_of_v<Deferred, T>, bool> = true>
typename T::future_type defer(Ts &&...args) {
  auto p = std::make_unique<T>(std::forward<Ts>(args)...);
  _dist(p.get());
  auto f = p->get_future().share();
  push_runable(std::move(p));
  return f;
}

// implementation details for deferring ops returning arrays
extern Deferred::future_type defer_array(Runable::ptr_type &&d,
                                          bool is_global);

// defer operations which do return a array, e.g. which are a Deferred
template <typename T, typename... Ts,
          std::enable_if_t<std::is_base_of_v<Deferred, T>, bool> = true>
Deferred::future_type defer(Ts &&...args) {
  return defer_array(std::move(std::make_unique<T>(std::forward<Ts>(args)...)),
                      true);
}

static void defer(nullptr_t) { push_runable(Runable::ptr_type()); }

struct UnDeferred : public Deferred {
  UnDeferred(array_i::ptr_type ptr) { set_value(std::move(ptr)); }

  void run() {}

  FactoryId factory() const {
    throw(std::runtime_error("No Factory for Undeferred."));
  }
};

template <typename L> struct DeferredLambda : public Runable {
  using promise_type = int;
  using future_type = int;

  L _l;

  DeferredLambda(L l) : _l(l) {}

  void run() { _l(); }

  bool generate_mlir(::mlir::OpBuilder &, const ::mlir::Location &,
                     jit::DepManager &) {
    return _l();
  }

  FactoryId factory() const {
    throw(std::runtime_error("No Factory for DeferredLambda."));
  }
};

template <typename L> void defer_lambda(L &&l) {
  push_runable(std::move(std::make_unique<DeferredLambda<L>>(l)));
}
} // namespace SHARPY
