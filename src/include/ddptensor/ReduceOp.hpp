// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct ReduceOp {
  static ddptensor *op(ReduceOpId op, const ddptensor &a,
                       const dim_vec_type &dim);
};
