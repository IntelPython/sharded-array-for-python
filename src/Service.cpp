// SPDX-License-Identifier: BSD-3-Clause

/*
 Service operations, mostly used internally.
 Dropping/out-of-scoping arrays.
 Replicating arrays.
*/

#include "sharpy/Service.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/FutureArray.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/Registry.hpp"
#include "sharpy/TypeDispatch.hpp"
#include "sharpy/jit/mlir.hpp"
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

namespace SHARPY {

bool inited = false;
bool finied = false;

// **************************************************************************

struct DeferredService : public DeferredT<Service::service_promise_type,
                                          Service::service_future_type> {
  enum Op : int { DROP, RUN, SERVICE_LAST };

  id_type _a;
  Op _op;

  DeferredService(Op op = SERVICE_LAST) : _a(), _op(op) {}
  DeferredService(Op op, id_type id) : _a(id), _op(op) {}

  void run() override {
    switch (_op) {
    case RUN:
      set_value(true);
      break;
    default:
      throw(std::invalid_argument(
          "Execution of unkown service operation requested."));
    }
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    switch (_op) {
    case DROP: {
      // drop from dep manager
      dm.drop(_a);
      // and from registry
      Registry::del(_a);
      break;
    }
    case RUN:
      return true;
    default:
      throw(std::invalid_argument(
          "MLIR generation for unkown service operation requested."));
    }

    return false;
  }

  FactoryId factory() const override { return F_SERVICE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
    ser.template value<sizeof(_op)>(_op);
  }
};

// **************************************************************************

struct DeferredReplicate : public Deferred {
  id_type _a;

  DeferredReplicate() : _a() {}
  DeferredReplicate(const array_i::future_type &a) : _a(a.guid()) {}

  void run() override {
    const auto a = std::move(Registry::get(_a).get());
    auto ary = dynamic_cast<NDArray *>(a.get());
    if (!ary) {
      throw std::invalid_argument("Expected NDArray in replicate.");
    }
    ary->replicate();
    set_value(a);
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const override { return F_REPLICATE; }

  template <typename S> void serialize(S &ser) {
    ser.template value<sizeof(_a)>(_a);
  }
};

// **************************************************************************

void Service::drop(const id_type a) {
  if (inited) {
    defer<DeferredService>(DeferredService::DROP, a);
  }
}

Service::service_future_type Service::run() {
  return defer<DeferredService>(DeferredService::RUN);
}

FutureArray *Service::replicate(const FutureArray &a) {
  return new FutureArray(defer<DeferredReplicate>(a.get()));
}

FACTORY_INIT(DeferredService, F_SERVICE);
FACTORY_INIT(DeferredReplicate, F_REPLICATE);
} // namespace SHARPY
