// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct LinAlgOp
{
    static tensor_i::future_type vecdot(tensor_i::future_type & a, tensor_i::future_type & b, int axis);
};
