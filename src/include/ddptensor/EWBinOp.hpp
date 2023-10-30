// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's elementwise binary ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

namespace DDPT {
struct EWBinOp {
  static ddptensor *op(EWBinOpId op, const py::object &a, const py::object &b);
};
} // namespace DDPT
