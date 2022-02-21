// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct ReduceOp
{
    static tensor_i::ptr_type op(ReduceOpId op, tensor_i::ptr_type a, const dim_vec_type & dim);
};
