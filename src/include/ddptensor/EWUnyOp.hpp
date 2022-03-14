// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct EWUnyOp
{
    static ddptensor * op(EWUnyOpId op, const ddptensor & a);
};
