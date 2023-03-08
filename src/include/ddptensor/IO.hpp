// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation ddptensor I/O ops.
*/

#pragma once

#include "ddptensor/SetGetItem.hpp"

struct IO {
  static GetItem::py_future_type to_numpy(const ddptensor &a);
};
