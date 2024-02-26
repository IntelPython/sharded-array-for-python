// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's linalg  ops.
*/

#pragma once

#include "FutureArray.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct LinAlgOp {
  static FutureArray *vecdot(const FutureArray &a, const FutureArray &b,
                             int axis);
};
} // namespace SHARPY
