// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's inplace elementwise binary ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

namespace DDPT {
struct IEWBinOp {
  static ddptensor *op(IEWBinOpId op, ddptensor &a, const py::object &b);
};
} // namespace DDPT
