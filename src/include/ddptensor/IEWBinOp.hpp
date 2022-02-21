// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct IEWBinOp
{
    static void op(IEWBinOpId op, tensor_i::ptr_type a, py::object & b);
};
