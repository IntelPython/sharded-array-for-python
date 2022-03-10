// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"

struct Creator
{
    static tensor_i::future_type create_from_shape(CreatorId op, const shape_type & shape, DTypeId dtype=FLOAT64);
    static tensor_i::future_type full(const shape_type & shape, const py::object & val, DTypeId dtype=FLOAT64);
    static tensor_i::future_type arange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype=INT64);
    static tensor_i::future_type mk_future(const py::object & b);
};
