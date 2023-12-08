// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's manipulation ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "FutureArray.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct ManipOp {
  static FutureArray *reshape(const FutureArray &a, const shape_type &shape,
                            const py::object &copy);
};
} // namespace SHARPY
