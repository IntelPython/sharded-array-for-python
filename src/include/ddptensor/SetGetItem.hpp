// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "tensor_i.hpp"
#include "p2c_ids.hpp"

struct GetItem
{
    static tensor_i::ptr_type __getitem__(tensor_i::ptr_type a, const std::vector<py::slice> & v);
    static py::object get_slice(tensor_i::ptr_type a, const std::vector<py::slice> & v);
};

struct SetItem
{
    static void __setitem__(tensor_i::ptr_type a, const std::vector<py::slice> & v, tensor_i::ptr_type b);
};
