// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation FutureArray I/O ops.
*/

#pragma once

#include "sharpy/SetGetItem.hpp"
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

namespace SHARPY {
struct IO {
  static GetItem::py_future_type to_numpy(const FutureArray &a);
  static FutureArray *from_locals(const std::vector<py::array> &a);
};
} // namespace SHARPY
