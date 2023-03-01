// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct IEWBinOp {
  static ddptensor *op(IEWBinOpId op, ddptensor &a, const py::object &b);
};
