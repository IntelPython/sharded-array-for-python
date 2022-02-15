// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"

struct Random
{
    using ptr_type = tensor_i::ptr_type;

    static ptr_type rand(DType dtype, const shape_type & shp, const py::object & lower, const py::object & upper);
    static void seed(uint64_t s);
};
