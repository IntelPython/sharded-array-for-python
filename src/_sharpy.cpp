// SPDX-License-Identifier: BSD-3-Clause

/*
  A Distributed Data-Parallel array library for Python, following the array
  API.

  Actual computation gets deferred and jit-compiled using MLIR.
  pybind11 handles the bridge to Python.

  We bridge dynamic dtypes of the Python array through dynamic type dispatch
  (TypeDispatch). This means the compiler will instantiate the full
  functionality for all elements types. Within kernels we dispatch the operation
  type by enum values. array_i is an abstract class to hide the element type
  which of the actual array. The concrete array implementation (NDArray)
  stores the element type as a dynamic attribute and dispatches computation as
  needed.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sched.h>
#include <stdlib.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring _a

#define DEF_PY11_ENUMS // used in p2c_types.hpp

#include "sharpy/Creator.hpp"
#include "sharpy/Deferred.hpp"
#include "sharpy/EWBinOp.hpp"
#include "sharpy/EWUnyOp.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/IEWBinOp.hpp"
#include "sharpy/IO.hpp"
#include "sharpy/LinAlgOp.hpp"
#include "sharpy/MPIMediator.hpp"
#include "sharpy/MPITransceiver.hpp"
#include "sharpy/ManipOp.hpp"
#include "sharpy/Random.hpp"
#include "sharpy/ReduceOp.hpp"
#include "sharpy/Service.hpp"
#include "sharpy/SetGetItem.hpp"
#include "sharpy/Sorting.hpp"
#include "sharpy/itac.hpp"
#include "sharpy/jit/mlir.hpp"

#include <fstream>
#include <iostream>

namespace SHARPY {
// #########################################################################
// The following classes are wrappers bridging pybind11 defs to TypeDispatch

rank_type myrank() { return getTransceiver()->rank(); }

std::thread *pprocessor = nullptr;

extern bool inited;
extern bool finied;

void sync_promises() {
  int vtWaitSym, vtSHARPYClass;
  VT(VT_classdef, "sharpy", &vtSHARPYClass);
  VT(VT_funcdef, "wait", vtSHARPYClass, &vtWaitSym);
  VT(VT_begin, vtWaitSym);
  py::gil_scoped_release release;
  (void)Service::run().get();
  VT(VT_end, vtWaitSym);
}

// users currently need to call fini to make MPI terminate gracefully
void fini() {
  if (finied)
    return;
  sync_promises();
  {
    auto guids = Registry::get_all();
    for (auto id : guids) {
      Service::drop(id);
    }
  }
  sync_promises();
  py::gil_scoped_release release;
  fini_mediator(); // stop task is sent in here
  if (pprocessor) {
    if (getTransceiver()->nranks() == 1)
      defer(nullptr);
    pprocessor->join();
    delete pprocessor;
    pprocessor = nullptr;
  }
  fini_transceiver();
  Deferred::fini();
  Registry::fini();
  jit::fini();
  inited = false;
  finied = true;
}

void init(bool cw, std::string libidtr) {
  if (inited)
    return;

  if (!std::ifstream(libidtr)) {
    throw std::runtime_error(std::string("Cannot find libidtr.so"));
  }

  init_transceiver(new MPITransceiver(cw));
  init_mediator(new MPIMediator());
  int cpu = sched_getcpu();
  std::cerr << "rank " << getTransceiver()->rank() << " is running on core "
            << cpu << std::endl;
  if (cw) {
    if (getTransceiver()->rank()) {
      process_promises(libidtr);
      fini();
      exit(0);
    }
  }
  pprocessor = new std::thread(process_promises, libidtr);
  inited = true;
  finied = false;
}

// #########################################################################

/// trigger compile&run and return python object
#define PY_SYNC_RETURN(_f)                                                     \
  {                                                                            \
    int vtWaitSym, vtSHARPYClass;                                              \
    VT(VT_classdef, "sharpy", &vtSHARPYClass);                                 \
    VT(VT_funcdef, "wait", vtSHARPYClass, &vtWaitSym);                         \
    VT(VT_begin, vtWaitSym);                                                   \
    py::handle res;                                                            \
    {                                                                          \
      py::gil_scoped_release release;                                          \
      Service::run();                                                          \
      res = (_f).get();                                                        \
    }                                                                          \
    VT(VT_end, vtWaitSym);                                                     \
    return py::reinterpret_steal<py::object>(res);                             \
  }

/// trigger compile&run and return given attribute _x
#define SYNC_RETURN(_f, _a)                                                    \
  int vtWaitSym, vtSHARPYClass;                                                \
  VT(VT_classdef, "sharpy", &vtSHARPYClass);                                   \
  VT(VT_funcdef, "wait", vtSHARPYClass, &vtWaitSym);                           \
  VT(VT_begin, vtWaitSym);                                                     \
  py::gil_scoped_release release;                                              \
  Service::run();                                                              \
  auto r = (_f).get().get() -> _a();                                           \
  VT(VT_end, vtWaitSym);                                                       \
  return r

/// Replicate sharpy/future and SYNC_RETURN attribute _a
#define REPL_SYNC_RETURN(_f, _a)                                               \
  auto r_ = std::unique_ptr<FutureArray>(Service::replicate(f));               \
  SYNC_RETURN(r_->get(), _a)

// Finally our Python module
PYBIND11_MODULE(_sharpy, m) {

  initFactories();

  jit::init();

  m.doc() = "A partitioned and distributed array";

  def_enums(m);
  py::enum_<_RANKS>(m, "_Ranks").value("_REPLICATED", REPLICATED);

  m.def("fini", &fini)
      .def("init", &init)
      .def("sync", &sync_promises)
      .def("myrank", &myrank)
      .def("_get_slice", &GetItem::get_slice)
      .def("_get_locals",
           [](const FutureArray &f, py::handle h) {
             PY_SYNC_RETURN(GetItem::get_locals(f, h));
           })
      .def("_from_locals", &IO::from_locals)
      .def("_gather",
           [](const FutureArray &f, rank_type root = REPLICATED) {
             PY_SYNC_RETURN(GetItem::gather(f, root));
           })
      .def("to_numpy",
           [](const FutureArray &f) { PY_SYNC_RETURN(IO::to_numpy(f)); });

  py::class_<Creator>(m, "Creator")
      .def("full", &Creator::full)
      .def("arange", &Creator::arange)
      .def("linspace", &Creator::linspace);

  py::class_<EWUnyOp>(m, "EWUnyOp").def("op", &EWUnyOp::op);
  py::class_<IEWBinOp>(m, "IEWBinOp").def("op", &IEWBinOp::op);
  py::class_<EWBinOp>(m, "EWBinOp").def("op", &EWBinOp::op);
  py::class_<ReduceOp>(m, "ReduceOp").def("op", &ReduceOp::op);
  py::class_<ManipOp>(m, "ManipOp").def("reshape", &ManipOp::reshape);
  py::class_<LinAlgOp>(m, "LinAlgOp").def("vecdot", &LinAlgOp::vecdot);

  py::class_<FutureArray>(m, "SHARPYFuture")
      // attributes we can get from the future itself
      .def_property_readonly(
          "dtype", [](const FutureArray &f) { return f.get().dtype(); })
      .def_property_readonly(
          "ndim", [](const FutureArray &f) { return f.get().rank(); })
      // attributes we can get from future without additional computation
      .def_property_readonly(
          "shape", [](const FutureArray &f) { SYNC_RETURN(f, shape); })
      .def_property_readonly("size",
                             [](const FutureArray &f) { SYNC_RETURN(f, size); })
      .def("__len__", [](const FutureArray &f) { SYNC_RETURN(f, __len__); })
      .def("__repr__", [](const FutureArray &f) { SYNC_RETURN(f, __repr__); })
      // attributes extracting values require replication
      .def("__bool__",
           [](const FutureArray &f) { REPL_SYNC_RETURN(f, __bool__); })
      .def("__float__",
           [](const FutureArray &f) { REPL_SYNC_RETURN(f, __float__); })
      .def("__int__",
           [](const FutureArray &f) { REPL_SYNC_RETURN(f, __int__); })
      .def("__index__",
           [](const FutureArray &f) { REPL_SYNC_RETURN(f, __int__); })
      // attributes returning a new FutureArray
      .def("astype", &ManipOp::astype)
      .def("to_device", &ManipOp::to_device)
      .def("__getitem__", &GetItem::__getitem__)
      .def("__setitem__", &SetItem::__setitem__)
      .def("map", &SetItem::map);
#undef REPL_SYNC_RETURN
#undef SYNC_RETURN

  py::class_<Random>(m, "Random")
      .def("seed", &Random::seed)
      .def("uniform", &Random::rand);

  // py::class_<dpdlpack>(m, "dpdlpack")
  //     .def("__dlpack__", &dpdlpack.__dlpack__);
}
} // namespace SHARPY
