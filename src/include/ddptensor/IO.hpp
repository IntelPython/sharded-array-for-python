// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"

struct IO {
  static py::object to_numpy(const ddptensor &a);
};
