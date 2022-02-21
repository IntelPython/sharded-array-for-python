// SPDX-License-Identifier: BSD-3-Clause

/*
  A Distributed Data-Parallel Tensor for Python, following the array API.
  
  XTensor handles the actual functionality on each process.
  pybind11 handles the bridge to Python.

  We bridge dynamic dtypes of the Python array through dynamic type dispatch (TypeDispatch).
  This means the compiler will instantiate the full functionality for all elements types.
  Within kernels we dispatch the operation type by enum values (see x.hpp).
  tensor_i is an abstract class to hide the element type which of the actual tensor.
  The concrete tensor implementation (DPTensorX, x.hpp) requires the element type 
  as a template parameter.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring _a

#include "ddptensor/MPITransceiver.hpp"
#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/EWUnyOp.hpp"
#include "ddptensor/ReduceOp.hpp"
#include "ddptensor/ManipOp.hpp"
#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/Random.hpp"

// #########################################################################
// The following classes are wrappers bridging pybind11 defs to TypeDispatch

rank_type myrank()
{
    return theTransceiver->rank();
}

Transceiver * theTransceiver = nullptr;
Mediator * theMediator = nullptr;

// users currently need to call fini to make MPI terminate gracefully
void fini()
{
    delete theMediator;
    theMediator = nullptr;
    delete theTransceiver;
    theTransceiver = nullptr;
}

// #########################################################################
// Finally our Python module
PYBIND11_MODULE(_ddptensor, m) {
    theTransceiver = new MPITransceiver();
    theMediator = new MPIMediator();

    m.doc() = "A partitioned and distributed tensor";

    def_enums(m);

    m.def("fini", &fini)
        .def("myrank", &myrank)
        .def("_get_slice", &GetItem::get_slice);

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

    py::class_<tensor_i, tensor_i::ptr_type>(m, "DPTensorX")
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

    py::class_<Random>(m, "Random")
        .def("seed", &Random::seed)
        .def("uniform", &Random::rand);

#if 0
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
