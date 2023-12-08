// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's reduction ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "FutureArray.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct ReduceOp {
  static FutureArray *op(ReduceOpId op, const FutureArray &a,
                       const dim_vec_type &dim);
};
} // namespace SHARPY
