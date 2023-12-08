// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's elementwise unary ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "FutureArray.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct EWUnyOp {
  static FutureArray *op(EWUnyOpId op, const FutureArray &a);
};
} // namespace SHARPY
