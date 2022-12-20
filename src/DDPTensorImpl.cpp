// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i.
// Interfaces are based on shared_ptr<tensor_i>.

#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/CppTypes.hpp>

#include <algorithm>


DDPTensorImpl::DDPTensorImpl(DTypeId dtype, uint64_t ndims,
                void * allocated, void * aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides,
                uint64_t * gs_allocated, uint64_t * gs_aligned, uint64_t * lo_allocated, uint64_t * lo_aligned,
                rank_type owner)
    : _owner(owner),
      _allocated(allocated),
      _aligned(aligned),
      _sizes(new intptr_t[ndims]),
      _strides(new intptr_t[ndims]),
      _gs_allocated(gs_allocated),
      _gs_aligned(gs_aligned),
      _lo_allocated(lo_allocated),
      _lo_aligned(lo_aligned),
      _offset(offset),
      _ndims(ndims),
      _dtype(dtype)
{
    memcpy(_sizes, sizes, ndims*sizeof(*_sizes));
    memcpy(_strides, strides, ndims*sizeof(*_strides));
}

DDPTensorImpl::DDPTensorImpl(DTypeId dtype, const shape_type & shp, rank_type owner)
    : _owner(owner),
      _ndims(shp.size()),
      _dtype(dtype)
{
    alloc();

    intptr_t stride = 1;
    auto ndims = shp.size();
    assert(ndims <= 1);
    for(auto i=0; i<ndims; ++i) {
        _sizes[i] = shp[i];
        _strides[ndims-i-1] = stride;
        stride *= shp[i];
    }
}

DDPTensorImpl::ptr_type DDPTensorImpl::clone(bool copy)
{
    // FIXME memory leak
    auto nd = ndims();
    auto sz = size();
    auto esz = sizeof_dtype(dtype());
    auto bsz = sz * esz;
    auto allocated = new (std::align_val_t(esz)) char[bsz];
    auto aligned = allocated;
    if(copy) memcpy(aligned, _aligned, bsz);
    // FIXME jit returns private mem
    // memcpy(gs_aligned, _gs_aligned, nd*sizeof(*gs_aligned));
    // auto gs_allocated = new uint64_t[nd];
    // auto gs_aligned = gs_allocated;
    auto gs_allocated = _gs_allocated;
    auto gs_aligned = _gs_aligned;
    auto lo_allocated = new uint64_t[nd];
    auto lo_aligned = lo_allocated;
    memcpy(lo_aligned, _lo_aligned, nd*sizeof(*lo_aligned));

    // strides and sizes are allocated/copied in constructor
    return std::make_shared<DDPTensorImpl>(dtype(), nd, allocated, aligned, _offset, _sizes, _strides,
                                           gs_allocated, gs_aligned, lo_allocated, lo_aligned, owner());
}

void DDPTensorImpl::alloc()
{
    auto esz = sizeof_dtype(_dtype);
    _allocated = new (std::align_val_t(esz)) char[esz*size()];
    _aligned = _allocated;
    auto nds = ndims();
    _sizes = new intptr_t[nds];
    _strides = new intptr_t[nds];
    _offset = 0;
}

void * DDPTensorImpl::data()
{
    void * ret;
    dispatch(_dtype, _aligned, [this, &ret](auto * ptr) { ret = ptr + this->_offset; });
    return ret;
}

std::string DDPTensorImpl::__repr__() const
{
    const auto nd = ndims();
    std::ostringstream oss;
    oss << "ddptensor{gs=(";
    for(auto i=0; i<nd; ++i) oss << _gs_aligned[i] << (i==nd-1 ? "" : ", ");
    oss << "), loff=(";
    for(auto i=0; i<nd; ++i) oss << _lo_aligned[i] << (i==nd-1 ? "" : ", ");
    oss << "), lsz=(";
    for(auto i=0; i<nd; ++i) oss << _sizes[i] << (i==nd-1 ? "" : ", ");
    oss << "), str=(";
    for(auto i=0; i<nd; ++i) oss << _strides[i] << (i==nd-1 ? "" : ", ");
    oss << "), p=" << _allocated << ", poff=" << _offset << "}\n";

    dispatch(_dtype, _aligned, [this, nd, &oss](auto * ptr) {
        auto cptr = ptr + this->_offset;
        if(nd>0) printit(oss, 0, cptr);
        else oss << *cptr;
    });
    return oss.str();
}

bool DDPTensorImpl::__bool__() const
{
    if(! is_replicated())
        throw(std::runtime_error("Cast to scalar bool: tensor is not replicated"));

    bool res;
    dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<bool>(ptr[this->_offset]); });
    return res;
}

double DDPTensorImpl::__float__() const
{
    if(! is_replicated())
        throw(std::runtime_error("Cast to scalar float: tensor is not replicated"));

    double res;
    dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<double>(ptr[this->_offset]); });
    return res;
}

int64_t DDPTensorImpl::__int__() const
{
    if(! is_replicated())
        throw(std::runtime_error("Cast to scalar int: tensor is not replicated"));

    float res;
    dispatch(_dtype, _aligned, [this, &res](auto * ptr) { res = static_cast<float>(ptr[this->_offset]); });
    return res;
}

void DDPTensorImpl::bufferize(const NDSlice & slc, Buffer & buff) const
{
    // FIXME slices/strides
#if 0
    if(slc.size() <= 0) return;
    NDSlice lslice = NDSlice(slice().tile_shape()).slice(slc);
#endif
    assert(_strides[0] == 1);
    auto pos = buff.size();
    auto sz = size()*item_size();
    buff.resize(pos + sz);
    void * out = buff.data() + pos;
    dispatch(_dtype, _aligned, [this, sz, out](auto * ptr) { memcpy(out, ptr + this->_offset, sz); });
}

void DDPTensorImpl::add_to_args(std::vector<void*> & args, int ndims)
{
    assert(ndims == this->ndims());
    // local tensor first
    intptr_t * buff = new intptr_t[dtensor_sz(ndims)];
    buff[0] = reinterpret_cast<intptr_t>(_allocated);
    buff[1] = reinterpret_cast<intptr_t>(_aligned);
    buff[2] = static_cast<intptr_t>(_offset);
    memcpy(buff+3, _sizes, ndims*sizeof(intptr_t));
    memcpy(buff+3+ndims, _strides, ndims*sizeof(intptr_t));
    args.push_back(buff);
    // second the team
    args.push_back(reinterpret_cast<void*>(1));
    if(ndims > 0)
    // global shape third
    buff = new intptr_t[dtensor_sz(1)];
    buff[0] = reinterpret_cast<intptr_t>(_gs_allocated);
    buff[1] = reinterpret_cast<intptr_t>(_gs_aligned);
    buff[2] = 0;
    buff[3] = ndims;
    buff[4] = 1;
    args.push_back(buff);
    assert(5 == memref_sz(1));
    // local offsets last
    buff = new intptr_t[dtensor_sz(1)];
    buff[0] = reinterpret_cast<intptr_t>(_lo_allocated);
    buff[1] = reinterpret_cast<intptr_t>(_lo_aligned);
    buff[2] = 0;
    buff[3] = ndims;
    buff[4] = 1;
    args.push_back(buff);
}