// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's setitem and getitem features.
  Also adds SPMD-like access to data.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "FutureArray.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {

struct GetItem {
  using py_promise_type = std::promise<py::object>;
  using py_future_type = std::shared_future<py::object>;

  static FutureArray *__getitem__(const FutureArray &a,
                                const std::vector<py::slice> &v);
  static py::object get_slice(const FutureArray &a,
                              const std::vector<py::slice> &v);
  static py_future_type get_locals(const FutureArray &a, py::handle h);
  static py_future_type gather(const FutureArray &a, rank_type root);
};

struct SetItem {
  static FutureArray *__setitem__(FutureArray &a, const std::vector<py::slice> &v,
                                const py::object &b);
  static FutureArray *map(FutureArray &a, py::object &b);
};
} // namespace SHARPY
