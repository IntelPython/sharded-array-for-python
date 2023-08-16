// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's setitem and getitem features.
  Also adds SPMD-like access to data.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct GetItem {
  using py_promise_type = std::promise<py::object>;
  using py_future_type = std::shared_future<py::object>;

  static ddptensor *__getitem__(const ddptensor &a,
                                const std::vector<py::slice> &v);
  static py::object get_slice(const ddptensor &a,
                              const std::vector<py::slice> &v);
  static py_future_type get_locals(const ddptensor &a, py::handle h);
  static py_future_type gather(const ddptensor &a, rank_type root);
};

struct SetItem {
  static ddptensor *__setitem__(ddptensor &a, const std::vector<py::slice> &v,
                                const py::object &b);
  static ddptensor *map(ddptensor &a, py::object &b);
};
