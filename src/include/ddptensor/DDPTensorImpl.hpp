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
#include <sstream>
#include <memory>
#include <algorithm>


class DDPTensorImpl : public tensor_i
{
    mutable rank_type _owner;
    PVSlice _slice;
    void * _allocated = nullptr;
    void * _aligned = nullptr;
    intptr_t * _sizes = nullptr;
    intptr_t * _strides = nullptr;
    uint64_t _offset = 0;
    DTypeId _dtype = DTYPE_LAST;

public:
    using ptr_type = std::shared_ptr<DDPTensorImpl>;

    DDPTensorImpl(const DDPTensorImpl &) = delete;
    DDPTensorImpl(DDPTensorImpl &&) = default;

    DDPTensorImpl(DTypeId dtype, uint64_t rank, void * allocated, void * aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides, rank_type owner=NOOWNER)
        : _owner(owner),
          _slice(shape_type(rank ? rank : 1, rank ? sizes[0] : 1), static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
          _allocated(allocated),
          _aligned(aligned),
          _sizes(new intptr_t[rank]),
          _strides(new intptr_t[rank]),
          _offset(offset),
          _dtype(dtype)
    {
        assert(rank <= 1);
        assert(rank == 0 || strides[0] == 1);

        memcpy(_sizes, sizes, rank*sizeof(intptr_t));
        memcpy(_strides, strides, rank*sizeof(intptr_t));
    }

    DDPTensorImpl(DTypeId dtype, const shape_type & shp, rank_type owner=NOOWNER)
        : _owner(owner),
          _slice(shp, static_cast<int>(owner==REPLICATED ? NOSPLIT : 0)),
          _dtype(dtype)
    {
        alloc();

        intptr_t stride = 1;
        auto rank = shp.size();
        for(auto i=0; i<rank; ++i) {
            _sizes[i] = shp[i];
            _strides[rank-i-1] = stride;
            stride *= shp[i];
        }
    }

    // incomplete, useful for computing meta information
    DDPTensorImpl(const uint64_t * shape, uint64_t N, rank_type owner=NOOWNER)
        : _owner(owner),
          _slice(shape_type(shape, shape+N), static_cast<int>(owner==REPLICATED ? NOSPLIT : 0))
    {
    }

    // incomplete, useful for computing meta information
    DDPTensorImpl()
        : _owner(REPLICATED),
          _slice(shape_type(), static_cast<int>(NOSPLIT))
    {
    }

    void alloc()
    {
        auto esz = sizeof_dtype(_dtype);
        _allocated = new (std::align_val_t(esz)) char[esz*_slice.size()];
        _aligned = _allocated;
        auto rank = _slice.ndims();
        _sizes = new intptr_t[rank];
        _strides = new intptr_t[rank];
        _offset = 0;
    }

    ~DDPTensorImpl()
    {
        delete [] _sizes;
        delete [] _strides;
    }

    void * data()
    {
        void * ret;
        dispatch(_dtype, _aligned, [this, &ret](auto * ptr) { ret = ptr + this->_offset; });
        return ret;
    }

    bool is_sliced() const
    {
        return _slice.is_sliced();
    }

    virtual std::string __repr__() const
    {
        // FIXME srides
        const auto sz = _slice.size();
        std::ostringstream oss;

        dispatch(_dtype, _aligned, [this, sz, &oss](auto * ptr) {
            ptr += this->_offset;
            for(auto i=0; i<sz; ++i) {
                oss << ptr[i] << " ";
            }
        });
        oss << std::endl;
        return oss.str();
    }

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

    virtual bool __bool__() const
    {
        if(! is_replicated())
            throw(std::runtime_error("Cast to scalar bool: tensor is not replicated"));
        
        bool res;
        dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<bool>(ptr[this->_offset]); });
        return res;
    }

    virtual double __float__() const
    {
        if(! is_replicated())
            throw(std::runtime_error("Cast to scalar float: tensor is not replicated"));

        double res;
        dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<double>(ptr[this->_offset]); });
        return res;
    }

    virtual int64_t __int__() const
    {
        if(! is_replicated())
            throw(std::runtime_error("Cast to scalar int: tensor is not replicated"));

        float res;
        dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<float>(ptr[this->_offset]); });
        return res;
    }

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

    virtual void bufferize(const NDSlice & slc, Buffer & buff) const
    {
        // FIXME slices/strides
#if 0
        if(slc.size() <= 0) return;
        NDSlice lslice = NDSlice(slice().tile_shape()).slice(slc);
#endif

        auto pos = buff.size();
        auto sz = _slice.size()*item_size();
        buff.resize(pos + sz);
        void * out = buff.data() + pos;
        dispatch(_dtype, _aligned, [this, sz, out](auto * ptr) { memcpy(out, ptr + this->_offset, sz); });
    }

    virtual uint64_t store_memref(intptr_t * buff, int rank)
    {
        assert(rank == _slice.ndims() || (_slice.ndims() == 1 && _slice.size() == 1));
        buff[0] = reinterpret_cast<intptr_t>(_allocated);
        buff[1] = reinterpret_cast<intptr_t>(_aligned);
        buff[2] = static_cast<intptr_t>(_offset);
        memcpy(buff+3, _sizes, rank*sizeof(intptr_t));
        memcpy(buff+3+rank, _strides, rank*sizeof(intptr_t));
        return 3 + 2*rank;
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

#if 0
union replica_u {
    double   _float64;
    float    _float32;
    int64_t  _int64;
    int32_t  _int32;
    int16_t  _int16;
    int8_t   _int8;
    uint64_t _uint64;
    uint32_t _uint32;
    uint16_t _uint16;
    uint8_t  _uint8;
    bool     _bool;

    void set(DTypeId dt, const void * ptr)
    {
        switch(dt) {
        case FLOAT64:
            _float64 = *reinterpret_cast<double*>(ptr);
            break;
        case INT64:
            _int64 = *reinterpret_cast<int64_t*>(ptr);
            break;
        case FLOAT32:
            _float = *reinterpret_cast<float*>(ptr);
            break;
        case INT32:
            _int32 = *reinterpret_cast<int32_t*>(ptr);
            break;
        case INT16:
            _int16 = *reinterpret_cast<int16_t*>(ptr);
            break;
        case INT8:
            _int8 = *reinterpret_cast<int8_t*>(ptr);
            break;
        case UINT64:
            _uint64 = *reinterpret_cast<uint64_t*>(ptr);
            break;
        case UINT32:
            _uint32 = *reinterpret_cast<uint32_t*>(ptr);
            break;
        case UINT16:
            _uint16 = *reinterpret_cast<uint16_t*>(ptr);
            break;
        case UINT8:
            _uint8 = *reinterpret_cast<uint8_t*>(ptr);
            break;
        case BOOL:
            _bool = *reinterpret_cast<bool*>(ptr);
            break;
        default:
            throw std::runtime_error("unknown dtype");
        };
    }

    void get(DTypeId dt, void * ptr)
    {
        switch(dt) {
        case FLOAT64:
            *reinterpret_cast<double*>(ptr) = _float64;
            break;
        case INT64:
            *reinterpret_cast<int64_t*>(ptr) = _float32;
            break;
        case FLOAT32:
            *reinterpret_cast<float*>(ptr) = _int64;
            break;
        case INT32:
            *reinterpret_cast<int32_t*>(ptr) = _int32;
            break;
        case INT16:
            *reinterpret_cast<int16_t*>(ptr) = _int16;
            break;
        case INT8:
            *reinterpret_cast<int8_t*>(ptr) = _int8;
            break;
        case UINT64:
            *reinterpret_cast<uint64_t*>(ptr) = _uint64;
            break;
        case UINT32:
            *reinterpret_cast<uint32_t*>(ptr) = _uint32;
            break;
        case UINT16:
            *reinterpret_cast<uint16_t*>(ptr) = _uint16;
            break;
        case UINT8:
            *reinterpret_cast<uint8_t*>(ptr) = _uint8;
            break;
        case BOOL:
            *reinterpret_cast<bool*>(ptr) = _bool;
            break;
        default:
            throw std::runtime_error("unknown dtype");
        };
    }
};
#endif // if 0
