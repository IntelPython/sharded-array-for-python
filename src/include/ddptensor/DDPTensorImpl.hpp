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
#include <sstream>


class DDPTensorImpl : public tensor_i
{
    mutable rank_type _owner;
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
          _ndims(N)
    {
        assert(_ndims <= 1);
    }

    // incomplete, useful for computing meta information
    DDPTensorImpl()
        : _owner(REPLICATED)
    {
        assert(_ndims <= 1);
    }

    DDPTensorImpl::ptr_type clone(bool copy = true);

    void alloc();

    ~DDPTensorImpl()
    {
        delete [] _sizes;
        delete [] _strides;
    }

    void * data();

    bool is_sliced() const
    {
        assert(false);
        return false;
    }

    virtual std::string __repr__() const;

    virtual DTypeId dtype() const
    {
        return _dtype;
    }

    virtual const shape_type & shape() const
    {
        assert(false);
        static shape_type dmy;
        return dmy;
    }

    virtual int ndims() const
    {
        return _ndims;
    }

    virtual uint64_t size() const
    {
        switch(ndims()) {
            case 0 : return 1;
            case 1 : return *_sizes;
            default: return std::accumulate(_sizes, _sizes+ndims(), 1, std::multiplies<intptr_t>());
        }
    }

    friend struct Service;

    virtual bool __bool__() const;
    virtual double __float__() const;
    virtual int64_t __int__() const;

    virtual uint64_t __len__() const
    {
        return ndims() ? *_sizes : 0;
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

    template<typename T>
    void printit(std::ostringstream & oss, uint64_t d, T * cptr) const
    {
        auto stride = _strides[d];
        auto sz = _sizes[d];
        if(d==ndims()-1) {
            oss << "[";
            for(auto i=0; i<sz; ++i) {
                oss << cptr[i*stride];
                if(i<sz-1) oss << " ";
            }
            oss << "]";
        } else {
            oss << "[";
            for(auto i=0; i<sz; ++i) {
                if(i) for(auto x=0; x<=d; ++x) oss << " ";
                printit(oss, d+1, cptr);
                if(i<sz-1) oss << "\n";
                cptr += stride;
            }
            oss << "]";
        }
    }
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
