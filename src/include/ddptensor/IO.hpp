// SPDX-License-Identifier: BSD-3-Clause

/*
  C++ representation ddptensor I/O ops.
*/

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"

struct IO {
  static py::object to_numpy(const ddptensor &a);
};
