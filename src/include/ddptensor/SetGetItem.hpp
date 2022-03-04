// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct GetItem
{
    static tensor_i::future_type __getitem__(const tensor_i::future_type & a, const std::vector<py::slice> & v);
    static py::object get_slice(const tensor_i::future_type & a, const std::vector<py::slice> & v);
    static py::object get_local(const tensor_i::future_type & a, py::handle h);
};

struct SetItem
{
    static tensor_i::future_type __setitem__(tensor_i::future_type & a, const std::vector<py::slice> & v, const tensor_i::future_type & b);
};
