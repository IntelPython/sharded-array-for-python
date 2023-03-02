// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's sorting ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct SortOp {
  static ddptensor *sort(const ddptensor &a, bool descending = false);
};
