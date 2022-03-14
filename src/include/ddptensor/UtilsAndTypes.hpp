// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <numeric>
#include <cstring>

#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "p2c_ids.hpp"

using shape_type = std::vector<uint64_t>;
using dim_vec_type = std::vector<int>;
using rank_type = uint64_t;

using Buffer = std::vector<uint8_t>;
using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;
using InputAdapter = bitsery::InputBufferAdapter<Buffer>;
using Serializer = bitsery::Serializer<OutputAdapter>;
using Deserializer = bitsery::Deserializer<InputAdapter>;

enum : rank_type {
    NOOWNER    = std::numeric_limits<rank_type>::max(),
    REPLICATED = std::numeric_limits<rank_type>::max() - 1,
    _OWNER_END = std::numeric_limits<rank_type>::max() - 1
};

template<typename T> struct DTYPE {};
template<> struct DTYPE<double>   { constexpr static DTypeId value = FLOAT64; };
template<> struct DTYPE<float>    { constexpr static DTypeId value = FLOAT32; };
template<> struct DTYPE<int64_t>  { constexpr static DTypeId value = INT64; };
template<> struct DTYPE<int32_t>  { constexpr static DTypeId value = INT32; };
template<> struct DTYPE<int16_t>  { constexpr static DTypeId value = INT16; };
template<> struct DTYPE<int8_t>   { constexpr static DTypeId value = INT8; };
template<> struct DTYPE<uint64_t> { constexpr static DTypeId value = UINT64; };
template<> struct DTYPE<uint32_t> { constexpr static DTypeId value = UINT32; };
template<> struct DTYPE<uint16_t> { constexpr static DTypeId value = UINT16; };
template<> struct DTYPE<uint8_t>  { constexpr static DTypeId value = UINT8; };
template<> struct DTYPE<bool>     { constexpr static DTypeId value = BOOL; };

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

using RedOpType = ReduceOpId;

inline RedOpType red_op(const char * op)
{
    if(!strcmp(op, "max")) return MAX;
    if(!strcmp(op, "min")) return MIN;
    if(!strcmp(op, "sum")) return SUM;
    if(!strcmp(op, "prod")) return PROD;
    if(!strcmp(op, "mean")) return MEAN;
    if(!strcmp(op, "std")) return STD;
    if(!strcmp(op, "var")) return VAR;
    throw std::logic_error("unsupported reduction operation");
}

inline shape_type reduce_shape(const shape_type & shape, const dim_vec_type & dims)
{
    auto ssz = shape.size();
    auto nd = dims.size();

    if(nd == 0 || nd == ssz) return shape_type{};

    shape_type shp(ssz - nd);
    if(shp.size()) {
        int p = -1;
        for(auto i = 0; i < ssz; ++i) {
            if(std::find(dims.begin(), dims.end(), i) == dims.end()) {
                shp[++p] = shape[i];
            }
        }
    }
    return shp;
}

template<typename T>
typename T::value_type VPROD(const T & v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<typename T::value_type>());
}

template <typename V>
std::string to_string(const std::vector<V>& vals, char sep=' ') {
    std::string s = "{";
    if(!vals.empty()) {
        auto _x = vals.begin();
        s += std::to_string(*_x);
        for(++_x; _x != vals.end(); ++_x) s += sep + std::to_string(*_x);
    }
    s += "}";
    return s;
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

extern bool is_cw();
extern bool is_spmd();

using id_type = uint64_t;

enum FactoryId : int {
    F_ARANGE,
    F_FROMSHAPE,
    F_FULL,
    F_UNYOP,
    F_EWUNYOP,
    F_IEWBINOP,
    F_EWBINOP,
    F_REDUCEOP,
    F_MANIPOP,
    F_LINALGOP,
    F_GETITEM,
    F_SETITEM,
    F_RANDOM,
    F_SERVICE,
    FACTORY_LAST
};
