// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "CppTypes.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {

template <> inline bool SharedBaseObject<py::object>::needGIL() const {
  return true;
}
template <> inline bool SharedBaseObject<py::handle>::needGIL() const {
  return true;
}

template <typename T> py::object get_impl_dtype() {
  return get_impl_dtype(DTYPE<T>::value);
};

inline PyScalar mk_scalar(const py::object &b, DTypeId dtype) {
  PyScalar s;
  if (b.is_none()) {
    s._float = std::numeric_limits<double>::quiet_NaN();
    return s;
  }
  switch (dtype) {
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
    return TypeDispatch2<OpDispatch>(std::forward<Ts>(args)...,
_downcast<bool>(a_ptr));
    */
  default:
    throw std::invalid_argument("unknown dtype");
  }
}

template <typename T, typename SZ, typename EL>
py::tuple _make_tuple(const T &v, const SZ &sz, const EL &el) {
  const auto n = sz(v);
  switch (n) {
  case 0:
    return py::tuple();
  case 1:
    return py::make_tuple(el(v, 0));
  case 2:
    return py::make_tuple(el(v, 0), el(v, 1));
  case 3:
    return py::make_tuple(el(v, 0), el(v, 1), el(v, 2));
  case 4:
    return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3));
  case 5:
    return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4));
  case 6:
    return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4),
                          el(v, 5));
  case 7:
    return py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4),
                          el(v, 5), el(v, 6));
  default:
    auto tpl = py::make_tuple(el(v, 0), el(v, 1), el(v, 2), el(v, 3), el(v, 4),
                              el(v, 5), el(v, 6), el(v, 7));
    for (auto i = 8ul; i < n; ++i) {
      tpl += py::make_tuple(el(v, i));
    }
    return tpl;
  }
}

template <typename T> py::tuple _make_tuple(const std::vector<T> &v) {
  using V = std::vector<T>;
  return _make_tuple(
      v, [](const V &v) { return v.size(); },
      [](const V &v, int i) { return v[i]; });
}

template <typename T> py::tuple _make_tuple(const T ptr, size_t n) {
  return _make_tuple(
      ptr, [n](const T &) { return n; },
      [](const T &v, int i) { return v[i]; });
}

template <typename T> T to_native(const py::object &o) { return o.cast<T>(); }

inline void compute_slice(const py::slice &slc, uint64_t length,
                          int64_t &offset, int64_t &size, int64_t &stride) {
  int64_t stop = 0;
  slc.compute(length, &offset, &stop, &stride, &size);
}

#if 0
inline py::tuple _make_tuple(const NDSlice & v)
{
    using V = NDSlice;
    return _make_tuple(v, [](const V & v){return v.ndims();}, [](const V & v, int i){return v.dim(i).pyslice();});
}
#endif
} // namespace SHARPY
