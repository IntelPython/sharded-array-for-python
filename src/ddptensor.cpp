// SPDX-License-Identifier: BSD-3-Clause

/*
  A Distributed Data-Parallel Tensor library for Python, following the array
  API.

  Actual computation gets deferred and jit-compiled using MLIR.
  pybind11 handles the bridge to Python.

  We bridge dynamic dtypes of the Python array through dynamic type dispatch
  (TypeDispatch). This means the compiler will instantiate the full
  functionality for all elements types. Within kernels we dispatch the operation
  type by enum values. tensor_i is an abstract class to hide the element type
  which of the actual tensor. The concrete tensor implementation (DDPTensorImpl)
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

#include "ddptensor/Creator.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/EWUnyOp.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/IO.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/MPITransceiver.hpp"
#include "ddptensor/ManipOp.hpp"
#include "ddptensor/Random.hpp"
#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/Service.hpp"
#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/Sorting.hpp"
#include "ddptensor/jit/mlir.hpp"

// #########################################################################
// The following classes are wrappers bridging pybind11 defs to TypeDispatch

rank_type myrank() { return getTransceiver()->rank(); }

std::thread *pprocessor = nullptr;

extern bool inited;
extern bool finied;

// users currently need to call fini to make MPI terminate gracefully
void fini() {
  py::gil_scoped_release release;
  if (finied)
    return;
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
  inited = false;
  finied = true;
}

void init(bool cw) {
  if (inited)
    return;
  init_transceiver(new MPITransceiver(cw));
  init_mediator(new MPIMediator());
  int cpu = sched_getcpu();
  std::cerr << "rank " << getTransceiver()->rank() << " is running on core "
            << cpu << std::endl;
  if (cw) {
    if (getTransceiver()->rank()) {
      process_promises();
      fini();
      exit(0);
    }
  }
  pprocessor = new std::thread(process_promises);
  inited = true;
  finied = false;
}

void sync_promises() {
  py::gil_scoped_release release;
  (void)Service::run().get();
}

// #########################################################################

/// trigger compile&run and return future value
#define PY_SYNC_RETURN(_f)                                                     \
  py::gil_scoped_release release;                                              \
  Service::run();                                                              \
  return (_f).get()

/// trigger compile&run and return given attribute _x
#define SYNC_RETURN(_f, _a)                                                    \
  py::gil_scoped_release release;                                              \
  Service::run();                                                              \
  return (_f).get().get()->_a()

/// Rerplicate ddptensor/future and SYNC_RETURN attributre _a
#define REPL_SYNC_RETURN(_f, _a)                                               \
  auto r_ = std::unique_ptr<ddptensor>(Service::replicate(f));                 \
  SYNC_RETURN(r_->get(), _a)

// Finally our Python module
PYBIND11_MODULE(_ddptensor, m) {
  // Factory::init<F_UNYOP>();
  Factory::init<F_ARANGE>();
  Factory::init<F_EWBINOP>();
  Factory::init<F_EWUNYOP>();
  Factory::init<F_FULL>();
  Factory::init<F_GETITEM>();
  Factory::init<F_IEWBINOP>();
  Factory::init<F_LINALGOP>();
  Factory::init<F_LINSPACE>();
  Factory::init<F_MANIPOP>();
  Factory::init<F_RANDOM>();
  Factory::init<F_REDUCEOP>();
  Factory::init<F_SERVICE>();
  Factory::init<F_SETITEM>();
  Factory::init<F_GATHER>();
  Factory::init<F_GETITEM>();

  jit::init();

  m.doc() = "A partitioned and distributed tensor";

  def_enums(m);
  py::enum_<_RANKS>(m, "_Ranks").value("_REPLICATED", REPLICATED);

  m.def("fini", &fini)
      .def("init", &init)
      .def("sync", &sync_promises)
      .def("myrank", &myrank)
      .def("_get_slice", &GetItem::get_slice)
      .def("_get_local",
           [](const ddptensor &f, py::handle h) {
             PY_SYNC_RETURN(GetItem::get_local(f, h));
           })
      .def("_gather",
           [](const ddptensor &f, rank_type root = REPLICATED) {
             PY_SYNC_RETURN(GetItem::gather(f, root));
           })
      .def("to_numpy",
           [](const ddptensor &f) { PY_SYNC_RETURN(IO::to_numpy(f)); });

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

  py::class_<ddptensor>(m, "DDPTFuture")
      // attributes we can get from the future itself
      .def_property_readonly("dtype",
                             [](const ddptensor &f) { return f.get().dtype(); })
      .def_property_readonly("ndim",
                             [](const ddptensor &f) { return f.get().rank(); })
      // attributes we can get from future without additional computation
      .def_property_readonly("shape",
                             [](const ddptensor &f) { SYNC_RETURN(f, shape); })
      .def_property_readonly("size",
                             [](const ddptensor &f) { SYNC_RETURN(f, size); })
      .def("__len__", [](const ddptensor &f) { SYNC_RETURN(f, __len__); })
      .def("__repr__", [](const ddptensor &f) { SYNC_RETURN(f, __repr__); })
      // attributes extracting values require replication
      .def("__bool__",
           [](const ddptensor &f) { REPL_SYNC_RETURN(f, __bool__); })
      .def("__float__",
           [](const ddptensor &f) { REPL_SYNC_RETURN(f, __float__); })
      .def("__int__", [](const ddptensor &f) { REPL_SYNC_RETURN(f, __int__); })
      .def("__index__",
           [](const ddptensor &f) { REPL_SYNC_RETURN(f, __int__); })
      // attributes returning a new ddptensor
      .def("__getitem__", &GetItem::__getitem__)
      .def("__setitem__", &SetItem::__setitem__)
      .def("map", &SetItem::map);
#undef REPL_SYNC_RETURN
#undef SYNC_RETURN

  py::class_<Random>(m, "Random")
      .def("seed", &Random::seed)
      .def("uniform", &Random::rand);

#if 0
    py::class_<tensor_i, tensor_i::ptr_type>(m, "DDPTensorImpl")
        .def_property_readonly("dtype", &tensor_i::dtype)
        .def_property_readonly("shape", &tensor_i::shape)
        .def_property_readonly("size", &tensor_i::size)
        .def_property_readonly("ndim", &tensor_i::ndim)
        .def("__bool__", &tensor_i::__bool__)
        .def("__float__", &tensor_i::__float__)
        .def("__int__", &tensor_i::__int__)
        .def("__index__", &tensor_i::__int__)
        .def("__len__", &tensor_i::__len__)
        .def("__repr__", &tensor_i::__repr__)
        .def("__getitem__", &GetItem::__getitem__)
        .def("__setitem__", &SetItem::__setitem__);

    py::class_<dtensor>(m, "dtensor")
        .def(py::init<const shape_type &, DTypeId>())
        .def_property_readonly("dtype", &dtensor::dtype)
        .def_property_readonly("shape", &dtensor::shape)
        .def("__bool__", &dtensor::__bool__)
        .def("__int__", &dtensor::__int__)
        .def("__float__", &dtensor::__float__)
        .def("__len__", &dtensor::__len__)
        .def("__repr__", &dtensor::__repr__)
        .def("__getitem__", py::overload_cast<const std::vector<int64_t> &>(&dtensor::__getitem__))
        .def("__getitem__", py::overload_cast<const std::vector<py::slice> &>(&dtensor::__getitem__))
        .def("__getitem__", py::overload_cast<const py::slice &>(&dtensor::__getitem__))
        .def("__getitem__", py::overload_cast<int64_t>(&dtensor::__getitem__))
        .def("__setitem__", &dtensor::__setitem__)
        .def("get_slice", &dtensor::get_slice);
#endif
  // py::class_<dpdlpack>(m, "dpdlpack")
  //     .def("__dlpack__", &dpdlpack.__dlpack__);
}
