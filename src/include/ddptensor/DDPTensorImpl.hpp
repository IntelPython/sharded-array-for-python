// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i.
// Interfaces are based on shared_ptr<tensor_i>.

#pragma once

#include "PVSlice.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"
#include "TypeDispatch.hpp"

#include <cstring>
#include <type_traits>
#include <memory>


class DDPTensorImpl : public tensor_i
{
    mutable rank_type _owner;
    PVSlice _slice;
    void * _allocated = nullptr;
    void * _aligned = nullptr;
    intptr_t * _sizes = nullptr;
    intptr_t * _strides = nullptr;
    uint64_t * _gs_allocated = nullptr;
    uint64_t * _gs_aligned = nullptr;
    uint64_t * _lo_allocated = nullptr;
    uint64_t * _lo_aligned = nullptr;
    uint64_t _offset = 0;
    uint64_t _ndims = 0;
    DTypeId _dtype = DTYPE_LAST;

public:
    using ptr_type = std::shared_ptr<DDPTensorImpl>;

    DDPTensorImpl(const DDPTensorImpl &) = delete;
    DDPTensorImpl(DDPTensorImpl &&) = default;

    DDPTensorImpl(DTypeId dtype, uint64_t ndims,
                  void * allocated, void * aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides,
                  uint64_t * gs_allocated, uint64_t * gs_aligned, uint64_t * lo_allocated, uint64_t * lo_aligned,
                  rank_type owner=NOOWNER);

    DDPTensorImpl(DTypeId dtype, const shape_type & shp, rank_type owner=NOOWNER);

    // incomplete, useful for computing meta information
    DDPTensorImpl(const uint64_t * shape, uint64_t N, rank_type owner=NOOWNER)
        : _owner(owner),
          _slice(shape_type(shape, shape+N), static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
          _ndims(N)
    {
    }

    // incomplete, useful for computing meta information
    DDPTensorImpl()
        : _owner(REPLICATED),
          _slice(shape_type(), static_cast<int>(NOSPLIT))
    {
    }

    void alloc();

    ~DDPTensorImpl()
    {
        delete [] _sizes;
        delete [] _strides;
    }

    void * data();

    bool is_sliced() const
    {
        return _slice.is_sliced();
    }

    virtual std::string __repr__() const;

    virtual DTypeId dtype() const
    {
        return _dtype;
    }

    virtual const shape_type & shape() const
    {
        return _slice.shape();
    }

    virtual int ndim() const
    {
        return _slice.ndims();
    }

    virtual uint64_t size() const
    {
        return _slice.size();
    }

    friend struct Service;

    virtual bool __bool__() const;
    virtual double __float__() const;
    virtual int64_t __int__() const;

    virtual uint64_t __len__() const
    {
        return _slice.slice().dim(0).size();
    }

    const PVSlice & slice() const
    {
        return _slice;
    }

    bool has_owner() const
    {
        return _owner < _OWNER_END;
    }

    void set_owner(rank_type o) const
    {
        _owner = o;
    }

    rank_type owner() const
    {
        return _owner;
    }

    bool is_replicated() const
    {
        return _owner == REPLICATED;
    }

    virtual int item_size() const
    {
        return sizeof_dtype(_dtype);
    }

    virtual void bufferize(const NDSlice & slc, Buffer & buff) const;

    virtual void add_to_args(std::vector<void*> & args, int ndims);
};

template<typename ...Ts>
static typename DDPTensorImpl::ptr_type mk_tnsr(Ts&&... args)
{
    return std::make_shared<DDPTensorImpl>(std::forward<Ts>(args)...);
}

template<typename ...Ts>
static tensor_i::future_type mk_ftx(Ts&&... args)
{
    return UnDeferred(mk_tnsr(std::forward(args)...)).get_future();
}
