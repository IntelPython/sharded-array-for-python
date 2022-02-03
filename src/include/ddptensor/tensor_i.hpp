// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "UtilsAndTypes.hpp"

class NDSlice;
class PVSlice;

///
/// Abstract interface for a tensor implementation.
/// Used for type elimination so we can bridge dynamic array types in Python to C++.
///
class tensor_i
{
public:
    typedef std::shared_ptr<tensor_i> ptr_type;

    virtual ~tensor_i(){}

    virtual const PVSlice & pvslice() = 0;
    virtual void bufferize(const NDSlice & slice, Buffer & buff) = 0;
    virtual int item_size() const = 0;
    virtual uint64_t id() const = 0;
    
    virtual ptr_type __getitem__(const NDSlice & slice) const = 0;
    virtual void __setitem__(const NDSlice & slice, const ptr_type &) = 0;
    virtual std::string __repr__() const = 0;
    virtual const shape_type & shape() const = 0;
    virtual DType dtype() const = 0;
    virtual bool __bool__() const = 0;
    virtual double __float__() const = 0;
    virtual int64_t __int__() const = 0;

    virtual ptr_type _ew_op(const char * op, const char * mod, py::args args, const py::kwargs & kwargs) = 0;
    virtual ptr_type _ew_unary_op(const char * op, bool is_method) const = 0;
    virtual ptr_type _ew_binary_op(const char * op, const ptr_type & b, bool is_method) const = 0;
    virtual ptr_type _ew_binary_op(const char * op, const py::object & b, bool is_method) const = 0;
    virtual void _ew_binary_op_inplace(const char * op, const ptr_type & b) = 0;
    virtual void _ew_binary_op_inplace(const char * op, const py::object & b) = 0;
    virtual ptr_type _reduce_op(const char * op, const dim_vec_type & dims) const = 0;
};
