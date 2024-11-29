// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's elementwise binary ops.
*/

#pragma once

#include "FutureArray.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct EWBinOp {
  static FutureArray *op(EWBinOpId op, py::object &a, const py::object &b);
};
} // namespace SHARPY
