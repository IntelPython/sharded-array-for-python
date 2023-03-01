// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct EWBinOp {
  static ddptensor *op(EWBinOpId op, const py::object &a, const py::object &b);
};
