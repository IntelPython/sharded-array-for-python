// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's manipulation ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct ManipOp {
  static ddptensor *reshape(const ddptensor &a, const shape_type &shape,
                            const py::object &copy);
};
