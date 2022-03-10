// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"

struct Replicate
{
    static tensor_i::future_type replicate(const tensor_i::future_type & a);
};
