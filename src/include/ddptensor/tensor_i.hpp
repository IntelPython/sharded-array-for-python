// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "UtilsAndTypes.hpp"

class NDSlice;

///
/// Abstract interface for a tensor implementation.
/// Used to hide the element type so we can bridge dynamic array types in Python to C++.
///
class tensor_i
{
public:
    typedef std::shared_ptr<tensor_i> ptr_type;

    virtual ~tensor_i() {};
    virtual std::string __repr__() const = 0;
    virtual DTypeId dtype() const = 0;
    virtual shape_type shape() const = 0;
    virtual int ndim() const = 0;
    virtual uint64_t size() const = 0;
    virtual bool __bool__() const = 0;
    virtual double __float__() const = 0;
    virtual int64_t __int__() const = 0;
    virtual uint64_t __len__() const = 0;

    // store all elements for given slice contiguously into provided Buffer
    virtual void bufferize(const NDSlice & slice, Buffer & buff) const = 0;
    // size of a single element (in bytes)
    virtual int item_size() const = 0;
    // global id, as assigned by Mediator::register_array
    virtual uint64_t id() const = 0;
};
