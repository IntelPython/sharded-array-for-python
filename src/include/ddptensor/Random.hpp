// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"

struct Random
{
    using future_type = tensor_i::future_type;

    static future_type rand(DTypeId dtype, const shape_type & shp, const py::object & lower, const py::object & upper);
    static void seed(uint64_t s);
};
