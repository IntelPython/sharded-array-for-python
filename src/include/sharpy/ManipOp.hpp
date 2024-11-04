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

  static FutureArray *astype(const FutureArray &a, DTypeId dtype,
                             const py::object &copy);

  static FutureArray *to_device(const FutureArray &a,
                                const std::string &device);

  static FutureArray *permute_dims(const FutureArray &array,
                                   const shape_type &axes);
};
} // namespace SHARPY
