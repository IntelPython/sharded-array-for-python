// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's inplace elementwise binary ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "FutureArray.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct IEWBinOp {
  static FutureArray *op(IEWBinOpId op, FutureArray &a, const py::object &b);
};
} // namespace SHARPY
