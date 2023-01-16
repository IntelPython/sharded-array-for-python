// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <future>

#include "CppTypes.hpp"

class NDSlice;

///
/// Abstract interface for a tensor implementation.
/// Used to hide the element type so we can bridge dynamic array types in Python to C++.
///
class tensor_i
{
public:

    typedef std::shared_ptr<tensor_i> ptr_type;
    typedef std::promise<ptr_type> promise_type;

    class TFuture : public std::shared_future<tensor_i::ptr_type>
    {
        id_type _id = -1;
        DTypeId _dtype = DTYPE_LAST;
        int     _rank = -1;
        
    public:
        TFuture() = default;
        TFuture(const TFuture & f) = default;
        TFuture(std::shared_future<tensor_i::ptr_type> && f, id_type id, DTypeId dt, int rank)
            : std::shared_future<tensor_i::ptr_type>(std::move(f)),
            _id(id),
            _dtype(dt),
            _rank(rank)
        {}

        ~TFuture()
        {}
        
        /// @return globally unique id
        id_type id() const { return _id; }

        /// @return dtype of future tensor
        DTypeId dtype() const { return _dtype; }

        /// @return rank (number of dims) of future tensor
        int rank() const { return _rank; }
    };

    typedef TFuture future_type;

    virtual ~tensor_i() {};
    virtual std::string __repr__() const = 0;
    virtual DTypeId dtype() const = 0;
    virtual const shape_type & shape() const = 0;
    virtual int ndims() const = 0;
    virtual uint64_t size() const = 0;
    virtual bool __bool__() const = 0;
    virtual double __float__() const = 0;
    virtual int64_t __int__() const = 0;
    virtual uint64_t __len__() const = 0;

    // size of a single element (in bytes)
    virtual int item_size() const = 0;
    // store tensor information in form of corresponding jit::JIT::DistMemRefDescriptor
    // @return stored size in number of intptr_t
    virtual void add_to_args(std::vector<void*> & args, int ndims) = 0;
};

#if 0
template<typename S>
void serialize(S & ser, tensor_i::future_type & f)
{
    uint64_t id = f.id();
    ser.value8b(id);
    if constexpr (std::is_same<Deserializer, S>::value) {
        
    }
}
#endif
