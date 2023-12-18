// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's manipulation ops.
*/

#pragma once

#include "FutureArray.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"

namespace SHARPY {
struct ManipOp {
  static FutureArray *reshape(const FutureArray &a, const shape_type &shape,
                              const py::object &copy);
};

struct AsType {
  static FutureArray *astype(const FutureArray &a, DTypeId dtype,
                             const py::object &copy);
};
} // namespace SHARPY
