// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "PyTypes.hpp"
#include "ddptensor.hpp"

struct Random {
  static ddptensor *rand(DTypeId dtype, const shape_type &shp,
                         const py::object &lower, const py::object &upper);
  static void seed(uint64_t s);
};
