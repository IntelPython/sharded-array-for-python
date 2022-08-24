// SPDX-License-Identifier: BSD-3-Clause

/*
  A Distributed Data-Parallel Tensor for Python, following the array API.

  XTensor handles the actual functionality on each process.
  pybind11 handles the bridge to Python.

  We bridge dynamic dtypes of the Python array through dynamic type dispatch (TypeDispatch).
  This means the compiler will instantiate the full functionality for all elements types.
  Within kernels we dispatch the operation type by enum values.
  tensor_i is an abstract class to hide the element type which of the actual tensor.
  The concrete tensor implementation (DDPTensorImpl) requires the element type
  as a template parameter.
 */

#include <sched.h>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring _a

#include "ddptensor/MPITransceiver.hpp"
#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/EWUnyOp.hpp"
#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/ManipOp.hpp"
#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/Random.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/Service.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/IO.hpp"
#include "ddptensor/jit/mlir.hpp"

// #########################################################################
// The following classes are wrappers bridging pybind11 defs to TypeDispatch

rank_type myrank()
{
    return theTransceiver->rank();
}

Transceiver * theTransceiver = nullptr;
Mediator * theMediator = nullptr;
std::thread * pprocessor;
bool _is_cw = false;

bool is_cw()
{
    return _is_cw && theTransceiver->nranks() > 1;
}

bool is_spmd()
{
    return !_is_cw && theTransceiver->nranks() > 1;
}

bool inited = false;
bool finied = false;

// users currently need to call fini to make MPI terminate gracefully
void fini()
{
    if(finied) return;
    delete theMediator;  // stop task is sent in here
    theMediator = nullptr;
    if(pprocessor) {
        if(theTransceiver->nranks() == 1) defer(nullptr);
        pprocessor->join();
        delete pprocessor;
    }
    delete theTransceiver;
    theTransceiver = nullptr;
    Deferred::fini();
    Registry::fini();
    inited = false;
    finied = true;
}

void init(bool cw)
{
    if(inited) return;
    theTransceiver = new MPITransceiver();
    theMediator = new MPIMediator();
    int cpu = sched_getcpu();
    std::cerr << "rank " << theTransceiver->rank() << " is running on core " << cpu << std::endl;
    if(cw) {
        _is_cw = true;
        if(theTransceiver->rank()) {
            process_promises();
            fini();
            exit(0);
        }
    }
    pprocessor = new std::thread(process_promises);
    inited = true;
    finied = false;
}

// #########################################################################
// Finally our Python module
PYBIND11_MODULE(_ddptensor, m) {
    Factory::init<F_ARANGE>();
    Factory::init<F_FULL>();
    Factory::init<F_FROMSHAPE>();
    // Factory::init<F_UNYOP>();
    Factory::init<F_EWUNYOP>();
    Factory::init<F_IEWBINOP>();
    Factory::init<F_EWBINOP>();
    Factory::init<F_REDUCEOP>();
    Factory::init<F_MANIPOP>();
    Factory::init<F_LINALGOP>();
    Factory::init<F_GETITEM>();
    Factory::init<F_SETITEM>();
    Factory::init<F_RANDOM>();
    Factory::init<F_SERVICE>();
    Factory::init<F_TONUMPY>();

    jit::init();

    m.doc() = "A partitioned and distributed tensor";

    def_enums(m);
    py::enum_<_RANKS>(m, "_Ranks")
        .value("_REPLICATED", REPLICATED);

    m.def("fini", &fini)
        .def("init", &init)
        .def("sync", &sync)
        .def("myrank", &myrank)
        .def("_get_slice", &GetItem::get_slice)
        .def("_get_local", &GetItem::get_local)
        .def("_gather", &GetItem::gather)
        .def("to_numpy", &IO::to_numpy)
        .def("ttt", &jit::ttt);

    py::class_<Creator>(m, "Creator")
        .def("create_from_shape", &Creator::create_from_shape)
        .def("full", &Creator::full)
        .def("arange", &Creator::arange);

    py::class_<EWUnyOp>(m, "EWUnyOp")
        .def("op", &EWUnyOp::op);

    py::class_<IEWBinOp>(m, "IEWBinOp")
        .def("op", &IEWBinOp::op);

    py::class_<EWBinOp>(m, "EWBinOp")
        .def("op", &EWBinOp::op);

    py::class_<ReduceOp>(m, "ReduceOp")
        .def("op", &ReduceOp::op);

    py::class_<ManipOp>(m, "ManipOp")
        .def("reshape", &ManipOp::reshape);

    py::class_<LinAlgOp>(m, "LinAlgOp")
        .def("vecdot", &LinAlgOp::vecdot);

#define GET_REPL(_f) std::unique_ptr<ddptensor>(Service::replicate(f))->get().get()
    py::class_<ddptensor>(m, "DDPTFuture")
        .def_property_readonly("dtype", [](const ddptensor & f) { return f.get().get()->dtype(); })
        .def_property_readonly("shape", [](const ddptensor & f) { return f.get().get()->shape(); })
        .def_property_readonly("size", [](const ddptensor & f) { return f.get().get()->size(); })
        .def_property_readonly("ndim", [](const ddptensor & f) { return f.get().get()->ndim(); })
        .def("__bool__", [](const ddptensor & f) { return GET_REPL(f)->__bool__(); })
        .def("__float__", [](const ddptensor & f) { return GET_REPL(f)->__float__(); })
        .def("__int__", [](const ddptensor & f) { return GET_REPL(f)->__int__(); })
        .def("__index__", [](const ddptensor & f) { return GET_REPL(f)->__int__(); })
        .def("__len__", [](const ddptensor & f) { return f.get().get()->__len__(); })
        .def("__repr__", [](const ddptensor & f) { return f.get().get()->__repr__(); })
        .def("__getitem__", &GetItem::__getitem__)
        .def("__setitem__", &SetItem::__setitem__);
#undef GET_REPL

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
    //py::class_<dpdlpack>(m, "dpdlpack")
    //    .def("__dlpack__", &dpdlpack.__dlpack__);
}
