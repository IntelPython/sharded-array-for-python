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

extern void process_promises(const std::string &libidtr);

// interface for promises/tasks to generate MLIR or execute immediately.
struct Runable {
  using ptr_type = std::unique_ptr<Runable>;
  virtual ~Runable() {};
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
  virtual bool isDeleter() { return false; }
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
  DeferredT(DTypeId dt, shape_type &&shape, std::string &&device,
            std::string &&team, id_type guid = Registry::NOGUID)
      : P(guid, dt, std::forward<shape_type>(shape),
          std::forward<std::string>(device),
          team.empty() ? std::string() : getTransceiver()->mesh()),
        Runable() {}
  DeferredT(DTypeId dt, shape_type &&shape, const std::string &device,
            const std::string &team, id_type guid = Registry::NOGUID)
      : P(guid, dt, std::forward<shape_type>(shape), device,
          team.empty() ? std::string() : getTransceiver()->mesh()),
        Runable() {}
  DeferredT(DTypeId dt, const shape_type &shape = {},
            const std::string &device = {}, const std::string &team = {},
            id_type guid = Registry::NOGUID)
      : P(guid, dt, shape, device,
          team.empty() ? std::string() : getTransceiver()->mesh()),
        Runable() {}
};

/// Deferred operation returning/producing a array
/// holds a guid as well as shape, dtype, device and team of future array
class Deferred : public DeferredT<array_i::promise_type, array_i::future_type> {
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
extern Deferred::future_type defer_array(Runable::ptr_type &&d, bool is_global);

// defer operations which do return a array, e.g. which are a Deferred
template <typename T, typename... Ts,
          std::enable_if_t<std::is_base_of_v<Deferred, T>, bool> = true>
Deferred::future_type defer(Ts &&...args) {
  return defer_array(std::move(std::make_unique<T>(std::forward<Ts>(args)...)),
                     true);
}

[[maybe_unused]] static void defer(std::nullptr_t) {
  push_runable(Runable::ptr_type());
}

struct UnDeferred : public Deferred {
  UnDeferred(array_i::ptr_type ptr) { set_value(std::move(ptr)); }

  void run() override {}

  FactoryId factory() const override {
    throw(std::runtime_error("No Factory for Undeferred."));
  }
};

template <typename G, typename R, bool D = false>
struct DeferredLambda : public Runable {
  using promise_type = int;
  using future_type = int;

  G _g;
  R _r;

  DeferredLambda(G g, R r) : _g(g), _r(r) {}

  bool isDeleter() override { return D; }

  void run() override { _r(); }

  bool generate_mlir(::mlir::OpBuilder &b, const ::mlir::Location &l,
                     jit::DepManager &d) override {
    return _g(b, l, d);
  }

  FactoryId factory() const override {
    throw(std::runtime_error("No Factory for DeferredLambda."));
  }
};

template <typename G, typename R> void defer_lambda(G &&g, R &&r) {
  push_runable(std::move(std::make_unique<DeferredLambda<G, R, false>>(g, r)));
}
template <typename G, typename R> void defer_del_lambda(G &&g, R &&r) {
  push_runable(std::move(std::make_unique<DeferredLambda<G, R, true>>(g, r)));
}
} // namespace SHARPY
