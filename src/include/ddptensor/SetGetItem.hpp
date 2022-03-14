// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "ddptensor.hpp"
#include "p2c_ids.hpp"

struct GetItem
{
    static ddptensor * __getitem__(const ddptensor & a, const std::vector<py::slice> & v);
    static py::object get_slice(const ddptensor & a, const std::vector<py::slice> & v);
    static py::object get_local(const ddptensor & a, py::handle h);
};

struct SetItem
{
    static ddptensor * __setitem__(ddptensor & a, const std::vector<py::slice> & v, const ddptensor & b);
};
