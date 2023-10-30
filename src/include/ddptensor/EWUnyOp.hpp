// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's elementwise unary ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

namespace DDPT {
struct EWUnyOp {
  static ddptensor *op(EWUnyOpId op, const ddptensor &a);
};
} // namespace DDPT
