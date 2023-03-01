// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ddptensor.hpp"
#include "UtilsAndTypes.hpp"
#include "p2c_ids.hpp"

struct Creator
{
    static ddptensor * create_from_shape(CreatorId op, const shape_type & shape, DTypeId dtype=FLOAT64);
    static ddptensor * full(const shape_type & shape, const py::object & val, DTypeId dtype=FLOAT64);
    static ddptensor * arange(uint64_t start, uint64_t end, uint64_t step, DTypeId dtype=INT64, uint64_t team=0);
    static std::pair<ddptensor *, bool> mk_future(const py::object & b);
};
