// SPDX-License-Identifier: BSD-3-Clause

#pragma once

class ddptensor;

struct Service
{
    static ddptensor * replicate(const ddptensor & a);
    static void drop(const ddptensor & a);
};
