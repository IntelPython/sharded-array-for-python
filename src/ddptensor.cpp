// SPDX-License-Identifier: BSD-3-Clause

/*
  A Distributed Data-Parallel Tensor for Python, following the array API.
  We have a 3-level hierachy
    1. tensor_i: the abstract interface, not bound to types (like numpy)
    2. dtensor_impl: a typed template layer with the actual functionality
    3. dtensor: the PYthon API delegating to untyped tensor_i
  We use pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring _a

#include "ddptensor/ddptensor_impl.hpp"
#include "ddptensor/MPITransceiver.hpp"
#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/x.hpp"

/// Thensor which is closely following the Python API
class dtensor
{
public:
    typedef tensor_i::ptr_type ptr_type;

    dtensor(dtensor &&) = default;

    dtensor(const shape_type & shape, DType dt)
        : _tensor(create_dtensor(PVSlice(shape), shape, dt))
    {
    }

    dtensor(const shape_type & shape, const char * create, const char * mod, py::args args, const py::kwargs & kwargs)
        : _tensor(create_dtensor(PVSlice(shape), shape, create, mod, args, kwargs))
    {
    }

    dtensor(ptr_type && t)
        : _tensor(std::move(t))
    {
    }

    shape_type shape() const
    {
        return _tensor->shape();
    }

    DType dtype() const
    {
        return _tensor->dtype();
    }

    py::object get_slice(const std::vector<py::slice> & v)
    {
        return _tensor->get_slice(NDSlice(v));
    }
        
    dtensor __getitem__(const NDIndex & v)
    {
        return dtensor(_tensor->__getitem__(NDSlice(v)));
    }

    dtensor __getitem__(int64_t i)
    {
        return __getitem__(NDIndex(1,i));
    }
    
    dtensor __getitem__(const std::vector<py::slice> & v)
    {
        return dtensor(_tensor->__getitem__(NDSlice(v)));
    }
    
    dtensor __getitem__(const py::slice & s)
    {
        return dtensor(_tensor->__getitem__(NDSlice(std::vector<py::slice>(1, s))));
    }

    void __setitem__(const std::vector<py::slice> & v, const dtensor * ob)
    {
        //const dtensor * ob = b.cast<const dtensor*>();
       _tensor->__setitem__(NDSlice(v), ob->_tensor);
    }

    std::string __repr__() const
    {
        return _tensor->__repr__();
    }

    // "__array_namespace__",  # (self, /, *, api_version=None)
    // "__dlpack__",  # (self, /, *, stream=None)
    // "__dlpack_device__",  # (self, /)

    bool __bool__()
    {
        return _tensor->__bool__();
    }
    
    double __float__()
    {
        return _tensor->__float__();
    }

    int64_t __int__()
    {
        return _tensor->__int__();
    }

    uint64_t __len__()
    {
        auto shp = _tensor->shape();
        return shp.empty() ? 1 : shp[0];
    }

    ptr_type _tensor;
};

dtensor create(const shape_type & shape, const char * op, const char * mod, py::args args, const py::kwargs& kwargs)
{
    return dtensor(shape, op, mod, args, kwargs);
}

dtensor ew_op(const dtensor & a, const char * op, const char * mod, py::args args, const py::kwargs& kwargs)
{
    return dtensor(a._tensor->_ew_op(op, mod, args, kwargs));
}

dtensor ew_unary_op(const dtensor & a, const char * op, bool is_method)
{
    return dtensor(a._tensor->_ew_unary_op(op, is_method));
}

dtensor ew_binary_op(const dtensor & a, const char * op, const py::object & b, bool is_method)
{
    const dtensor * ob = nullptr;
    try {
        ob = b.cast<const dtensor*>();
    } catch(...) {
        return dtensor(a._tensor->_ew_binary_op(op, b, is_method));
    }
    return dtensor(a._tensor->_ew_binary_op(op, ob->_tensor, is_method));
}

dtensor & ew_binary_op_inplace(dtensor & a, const char * op, const py::object & b)
{
    const dtensor * ob = nullptr;
    try {
        ob = b.cast<const dtensor*>();
    } catch(...) {
        a._tensor->_ew_binary_op_inplace(op, b);
    }
    a._tensor->_ew_binary_op_inplace(op, ob->_tensor);
    return a;
}

dtensor reduce_op(const dtensor & a, const char * op, const py::kwargs & kwargs)
{
    dim_vec_type dims;
    if(kwargs.contains("axis")) {
        auto ax = kwargs["axis"];
        if(!ax.is_none()) {
            try {
                auto a = ax.cast<dim_vec_type::value_type>();
                dims.resize(1);
                dims[0] = a;
            } catch(...) {
                dims = ax.cast<dim_vec_type>();
            }
        }
    }
    return dtensor(a._tensor->_reduce_op(op, dims));
}

// ###################################################################
// ###################################################################
// ###################################################################

template<template<typename OD> class OpDispatch, typename... Ts>
auto TypeDispatch(DType dt, Ts&&... args)
{
    switch(dt) {
    case DT_FLOAT64:
        return OpDispatch<double>::op(std::forward<Ts>(args)...);
    case DT_FLOAT32:
        return OpDispatch<float>::op(std::forward<Ts>(args)...);
    case DT_INT64:
        return OpDispatch<int64_t>::op(std::forward<Ts>(args)...);
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

rank_type myrank()
{
    return theTransceiver->rank();
}

Transceiver * theTransceiver = nullptr;
Mediator * theMediator = nullptr;

void fini()
{
    delete theMediator;
    theMediator = nullptr;
    delete theTransceiver;
    theTransceiver = nullptr;
}
    
PYBIND11_MODULE(_ddptensor, m) {
    theTransceiver = new MPITransceiver();
    theMediator = new MPIMediator();

    m.doc() = "A partitioned and distributed tensor";

    py::class_<Creator>(m, "Creator")
        .def("create_from_shape", &Creator::create_from_shape)
        .def("full", &Creator::full);

    py::class_<IEWBinOp>(m, "IEWBinOp")
        .def("op", &IEWBinOp::op);

    py::class_<EWBinOp>(m, "EWBinOp")
        .def("op", &EWBinOp::op);

    py::class_<x::DPTensorBaseX, x::DPTensorBaseX::ptr_type>(m, "DPTensorX")
        .def("__repr__", &x::DPTensorBaseX::__repr__);

    def_enums(m);

    py::enum_<DType>(m, "dtype")
        .value("float64", DT_FLOAT64)
        .value("int64", DT_INT64)
        .value("bool", DT_BOOL)
        .export_values();

    m.def("fini", &fini);
    m.def("myrank", &myrank);

    m.def("create", &create);
    m.def("ew_op", &ew_op);
    m.def("ew_unary_op", &ew_unary_op);
    m.def("ew_binary_op", &ew_binary_op);
    m.def("ew_binary_op_inplace", &ew_binary_op_inplace);
    m.def("reduce_op", &reduce_op);

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

    //py::class_<dpdlpack>(m, "dpdlpack")
    //    .def("__dlpack__", &dpdlpack.__dlpack__);
}
