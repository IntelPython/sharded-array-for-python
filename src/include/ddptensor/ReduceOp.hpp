// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's reduction ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct ReduceOp {
  static ddptensor *op(ReduceOpId op, const ddptensor &a,
                       const dim_vec_type &dim);
};
