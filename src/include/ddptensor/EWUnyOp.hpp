// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct EWUnyOp
{
    static tensor_i::future_type op(EWUnyOpId op, const tensor_i::future_type & a);
};
