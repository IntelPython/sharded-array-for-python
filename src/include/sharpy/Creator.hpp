// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's creation functions.
*/

#pragma once

#include "FutureArray.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"
#include <string>

namespace SHARPY {

struct Creator {
  static FutureArray *full(const shape_type &shape, const py::object &val,
                           DTypeId dtype, const std::string &device,
                           const std::string &team);
  static FutureArray *arange(uint64_t start, uint64_t end, uint64_t step,
                             DTypeId dtype, const std::string &device,
                             const std::string &team);
  static FutureArray *linspace(double start, double end, uint64_t num,
                               bool endpoint, DTypeId dtype,
                               const std::string &device,
                               const std::string &team);
  static std::pair<FutureArray *, bool> mk_future(const py::object &b,
                                                  const std::string &device,
                                                  const std::string &team,
                                                  DTypeId dtype);
};
} // namespace SHARPY
