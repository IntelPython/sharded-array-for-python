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
#include "ddptensor/x.hpp"

// Dependent on dt, dispatch arguments to a operation class.
// The operation must
//    * be a template class accepting the element type as argument
//    * implement one or more "op" methods matching the given arguments (args)
// All arguments other than dt are opaquely passed to the operation.
template<template<typename OD> class OpDispatch, typename... Ts>
auto TypeDispatch(DType dt, Ts&&... args)
{
    switch(dt) {
    case DT_FLOAT64:
        return OpDispatch<double>::op(std::forward<Ts>(args)...);
    case DT_INT64:
        return OpDispatch<int64_t>::op(std::forward<Ts>(args)...);
    case DT_FLOAT32:
        return OpDispatch<float>::op(std::forward<Ts>(args)...);
    case DT_INT32:
        return OpDispatch<int32_t>::op(std::forward<Ts>(args)...);
    case DT_INT16:
        return OpDispatch<int16_t>::op(std::forward<Ts>(args)...);
    case DT_UINT64:
        return OpDispatch<uint64_t>::op(std::forward<Ts>(args)...);
    case DT_UINT32:
        return OpDispatch<uint32_t>::op(std::forward<Ts>(args)...);
    case DT_UINT16:
        return OpDispatch<uint16_t>::op(std::forward<Ts>(args)...);
        /* FIXME
    case DT_BOOL:
        return OpDispatch<bool>::op(std::forward<Ts>(args)...);
        */
    default:
        throw std::runtime_error("unknown dtype");
    }
}

// #########################################################################
// The following classes are wrappers bridging pybind11 defs to TypeDispatch

struct Creator
{

    static auto create_from_shape(CreatorId op, shape_type && shape, DType dtype=DT_FLOAT64)
    {
        return TypeDispatch<x::Creator>(dtype, op, std::forward<shape_type>(shape));
    }

    static auto full(shape_type && shape, py::object && val, DType dtype=DT_FLOAT64)
    {
        auto op = FULL;
        return TypeDispatch<x::Creator>(dtype, op, std::forward<shape_type>(shape), std::forward<py::object>(val));
    }
};

struct IEWBinOp
{
    static auto op(IEWBinOpId op, x::DPTensorBaseX::ptr_type a, x::DPTensorBaseX::ptr_type b)
    {
        return TypeDispatch<x::IEWBinOp>(a->dtype(), op, a, b);
    }
};

struct EWBinOp
{
    static auto op(EWBinOpId op, x::DPTensorBaseX::ptr_type a, x::DPTensorBaseX::ptr_type b)
    {
        return TypeDispatch<x::EWBinOp>(a->dtype(), op, a, b);
    }
};

struct EWUnyOp
{
    static auto op(EWUnyOpId op, x::DPTensorBaseX::ptr_type a)
    {
        return TypeDispatch<x::EWUnyOp>(a->dtype(), op, a);
    }
};

struct ReduceOp
{
    static auto op(ReduceOpId op, x::DPTensorBaseX::ptr_type a, const dim_vec_type & dim)
    {
        return TypeDispatch<x::ReduceOp>(a->dtype(), op, a, dim);
    }
};

struct GetItem
{
    static auto __getitem__(x::DPTensorBaseX::ptr_type a, const std::vector<py::slice> & v)
    {
        return TypeDispatch<x::GetItem>(a->dtype(), a, NDSlice(v));
    }
    static auto get_slice(x::DPTensorBaseX::ptr_type a, const std::vector<py::slice> & v)
    {
        return TypeDispatch<x::SPMD>(a->dtype(), a, NDSlice(v));
    }
};

struct SetItem
{
    static auto __setitem__(x::DPTensorBaseX::ptr_type a, const std::vector<py::slice> & v, x::DPTensorBaseX::ptr_type b)
    {
        return TypeDispatch<x::SetItem>(a->dtype(), a, NDSlice(v), b);
    }
};

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

    py::enum_<DType>(m, "dtype")
        .value("float64", DT_FLOAT64)
        .value("int64", DT_INT64)
        .value("bool", DT_BOOL)
        .export_values();

    m.def("fini", &fini)
        .def("myrank", &myrank)
        .def("_get_slice", &GetItem::get_slice);

    py::class_<Creator>(m, "Creator")
        .def("create_from_shape", &Creator::create_from_shape)
        .def("full", &Creator::full);

    py::class_<EWUnyOp>(m, "EWUnyOp")
        .def("op", &EWUnyOp::op);

    py::class_<IEWBinOp>(m, "IEWBinOp")
        .def("op", &IEWBinOp::op);

    py::class_<EWBinOp>(m, "EWBinOp")
        .def("op", &EWBinOp::op);

    py::class_<ReduceOp>(m, "ReduceOp")
        .def("op", &ReduceOp::op);

    py::class_<x::DPTensorBaseX, x::DPTensorBaseX::ptr_type>(m, "DPTensorX")
        .def_property_readonly("dtype", &x::DPTensorBaseX::dtype)
        .def_property_readonly("shape", &x::DPTensorBaseX::shape)
        .def_property_readonly("size", &x::DPTensorBaseX::size)
        .def_property_readonly("ndim", &x::DPTensorBaseX::ndim)
        .def("__bool__", &x::DPTensorBaseX::__bool__)
        .def("__float__", &x::DPTensorBaseX::__float__)
        .def("__int__", &x::DPTensorBaseX::__int__)
        .def("__index__", &x::DPTensorBaseX::__int__)
        .def("__len__", &x::DPTensorBaseX::__len__)
        .def("__repr__", &x::DPTensorBaseX::__repr__)
        .def("__getitem__", &GetItem::__getitem__)
        .def("__setitem__", &SetItem::__setitem__);

#if 0
    py::class_<dtensor>(m, "dtensor")
        .def(py::init<const shape_type &, DType>())
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
