// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "p2c_ids.hpp"

template<typename T> py::object get_impl_dtype() { return get_impl_dtype(DTYPE<T>::value); };

union PyScalar
{
    int64_t _int;
    double _float;
};

inline PyScalar mk_scalar(const py::object & b, DTypeId dtype)
{
    PyScalar s;
    switch(dtype) {
    case FLOAT64:
    case FLOAT32:
        s._float = b.cast<double>();
        return s;
    case INT64:
    case INT32:
    case INT16:
    case INT8:
    case UINT64:
    case UINT32:
    case UINT16:
    case UINT8:
        s._int = b.cast<int64_t>();
        return s;
        /* FIXME
    case BOOL:
        return TypeDispatch2<OpDispatch>(std::forward<Ts>(args)..., _downcast<bool>(a_ptr));
        */
    default:
        throw std::runtime_error("unknown dtype");
    }
}

inline py::module_ get_array_impl(const py::object & = py::none())
{   // FIXME ary.attr("__array_namespace__")();
    static const char * _array_impl = nullptr;
    if(_array_impl == nullptr) {
        _array_impl = getenv("DDPNP_ARRAY");
        if(_array_impl == nullptr) _array_impl = "numpy";
    }
    py::module_ _array_ns = py::module_::import(_array_impl);
    return _array_ns;
}

inline const py::object & get_impl_dtype(const DTypeId dt)
{
    static py::object _dtypes [DTYPE_LAST] {py::none()};
    if(_dtypes[FLOAT32].is(py::none())) {
        auto mod = get_array_impl();
        _dtypes[FLOAT32] = mod.attr("float32");
        _dtypes[FLOAT64] = mod.attr("float64");
        _dtypes[INT16] = mod.attr("int16");
        _dtypes[INT32] = mod.attr("int32");
        _dtypes[INT64] = mod.attr("int64");
#if 0 // FIXME torch
        _dtypes[UINT16] = mod.attr("uint16");
        _dtypes[UINT32] = mod.attr("uint32");
        _dtypes[UINT64] = mod.attr("uint64");
#endif
        _dtypes[BOOL] = mod.attr("bool");
    }
    return _dtypes[dt];
}

template<typename T, typename SZ, typename EL>
py::tuple _make_tuple(const T & v, const SZ & sz, const EL & el)
{
    const auto n = sz(v);
    switch(n) {
    case 0: return py::tuple();
    case 1: return py::make_tuple(el(v, 0));
    case 2: return py::make_tuple(el(v, 0), el(v, 1));
    case 3: return py::make_tuple(el(v, 0), el(v, 1), el(v, 2));
    case 4: return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3));
    case 5: return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4));
    case 6: return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4), el(v, 5));
    case 7: return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4), el(v, 5), el(v, 6));
    default:
        auto tpl = py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4), el(v, 5), el(v, 6), el(v, 7));
        for(auto i = 8; i < n; ++i) {
            tpl += py::make_tuple(el(v, i));
        }
        return tpl;
    }
}

template<typename T>
py::tuple _make_tuple(const std::vector<T> & v)
{
    using V = std::vector<T>;
    return _make_tuple(v, [](const V & v){return v.size();}, [](const V & v, int i){return v[i];});
}

template<typename T>
T to_native(const py::object & o)
{
    return o.cast<T>();
}

inline void compute_slice(const py::slice & slc, uint64_t & offset, uint64_t & size, uint64_t & stride)
{
    uint64_t dmy = 0;
    slc.compute(std::numeric_limits<int64_t>::max(), &offset, &dmy, &stride, &size);
}

#if 0
inline py::tuple _make_tuple(const NDSlice & v)
{
    using V = NDSlice;
    return _make_tuple(v, [](const V & v){return v.ndims();}, [](const V & v, int i){return v.dim(i).pyslice();});
}
#endif
