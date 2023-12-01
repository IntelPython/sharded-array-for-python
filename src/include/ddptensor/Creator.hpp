// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation of the array-API's creation functions.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"
#include <string>

namespace DDPT {

struct Creator {
  static ddptensor *full(const shape_type &shape, const py::object &val,
                         DTypeId dtype, const std::string &device,
                         uint64_t team);
  static ddptensor *arange(uint64_t start, uint64_t end, uint64_t step,
                           DTypeId dtype, const std::string &device,
                           uint64_t team);
  static ddptensor *linspace(double start, double end, uint64_t num,
                             bool endpoint, DTypeId dtype,
                             const std::string &device, uint64_t team);
  static std::pair<ddptensor *, bool> mk_future(const py::object &b,
                                                const std::string &device,
                                                uint64_t team, DTypeId dtype);
};
} // namespace DDPT
