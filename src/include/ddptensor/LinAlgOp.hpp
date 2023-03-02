// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's linalg  ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct LinAlgOp {
  static ddptensor *vecdot(const ddptensor &a, const ddptensor &b, int axis);
};
