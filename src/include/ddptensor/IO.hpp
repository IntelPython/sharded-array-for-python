// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation ddptensor I/O ops.
*/

#pragma once

#include "ddptensor/SetGetItem.hpp"
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

struct IO {
  static GetItem::py_future_type to_numpy(const ddptensor &a);
  static ddptensor *from_locals(const std::vector<py::array> &a);
};
