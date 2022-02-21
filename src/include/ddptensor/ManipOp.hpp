// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct ManipOp
{
    static tensor_i::ptr_type reshape(tensor_i::ptr_type a, const shape_type & shape);
};
