// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i.
// Interfaces are based on shared_ptr<tensor_i>.

#include <ddptensor/CppTypes.hpp>
#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/Transceiver.hpp>

#include <algorithm>
#include <iostream>

DDPTensorImpl::DDPTensorImpl(
    Transceiver *transceiver, DTypeId dtype, shape_type gShape,
    void *l_allocated, void *l_aligned, intptr_t l_offset,
    const intptr_t *l_sizes, const intptr_t *l_strides, void *o_allocated,
    void *o_aligned, intptr_t o_offset, const intptr_t *o_sizes,
    const intptr_t *o_strides, void *r_allocated, void *r_aligned,
    intptr_t r_offset, const intptr_t *r_sizes, const intptr_t *r_strides,
    uint64_t *lo_allocated, uint64_t *lo_aligned, rank_type owner)
    : _owner(owner), _transceiver(transceiver), _gShape(gShape),
      _lo_allocated(lo_allocated), _lo_aligned(lo_aligned),
      _lhsHalo(gShape.size(), l_allocated, l_aligned, l_offset, l_sizes,
               l_strides),
      _lData(gShape.size(), o_allocated, o_aligned, o_offset, o_sizes,
             o_strides),
      _rhsHalo(gShape.size(), r_allocated, r_aligned, r_offset, r_sizes,
               r_strides),
      _dtype(dtype) {
  if (ndims() == 0) {
    _owner = REPLICATED;
  }
  assert(!_transceiver || _transceiver == getTransceiver());
}

DDPTensorImpl::DDPTensorImpl(DTypeId dtype, const shape_type &shp,
                             rank_type owner)
    : _owner(owner), _gShape(shp), _dtype(dtype) {

  auto esz = sizeof_dtype(_dtype);
  auto lsz =
      std::accumulate(shp.begin(), shp.end(), esz, std::multiplies<intptr_t>());
  auto allocated = aligned_alloc(esz, lsz);
  auto nds = ndims();
  auto sizes = new intptr_t[nds];
  auto strides = new intptr_t[nds];
  intptr_t stride = 1;
  assert(nds <= 1);
  for (auto i = 0; i < nds; ++i) {
    sizes[i] = shp[i];
    strides[nds - i - 1] = stride;
    stride *= shp[i];
  }
  _lData = DynMemRef(nds, allocated, allocated, 0, sizes, strides);
  assert(!_transceiver || _transceiver == getTransceiver());
}

// incomplete, useful for computing meta information
DDPTensorImpl::DDPTensorImpl(const int64_t *shape, uint64_t N, rank_type owner)
    : _owner(owner), _gShape(shape, shape + N) {
  assert(ndims() <= 1);
  assert(!_transceiver || _transceiver == getTransceiver());
}

// from numpy
DDPTensorImpl::DDPTensorImpl(DTypeId dtype, ssize_t ndims, const ssize_t *shape,
                             const intptr_t *strides, void *data)
    : _owner(NOOWNER), _gShape(shape, shape + ndims),
      _lo_allocated(
          static_cast<uint64_t *>(calloc(ndims, sizeof_dtype(dtype)))),
      _lo_aligned(_lo_allocated),
      _lData(ndims, data, data, 0, reinterpret_cast<const intptr_t *>(shape),
             reinterpret_cast<const intptr_t *>(strides)),
      _dtype(dtype) {}

void DDPTensorImpl::set_base(const tensor_i::ptr_type &base) {
  _base = new SharedBaseObject<tensor_i::ptr_type>(base);
}
void DDPTensorImpl::set_base(BaseObj *obj) { _base = obj; }

DDPTensorImpl::~DDPTensorImpl() {
  if (!_base) {
    // FIXME it seems possible that halos get reallocated even with when there
    // is a base
    if (_lhsHalo._allocated != _rhsHalo._allocated)
      _lhsHalo.freeData(); // lhs and rhs can be identical
    _lData.freeData();
    _rhsHalo.freeData();
  }
  free(_lo_allocated);
  delete _base;
}

void *DDPTensorImpl::data() {
  void *ret;
  dispatch(_dtype, _lData._aligned,
           [this, &ret](auto *ptr) { ret = ptr + this->_lData._offset; });
  return ret;
}

bool DDPTensorImpl::is_sliced() const {
  if (ndims() == 0)
    return false;
  auto d = ndims() - 1;
  intptr_t tsz = _lData._strides[d];
  if (tsz == 1) {
    for (; d > 0; --d) {
      tsz *= _lData._sizes[d];
      if (tsz <= 0)
        break;
      if (_lData._strides[d - 1] > tsz)
        return true;
    }
  }
  return false;
}

std::string DDPTensorImpl::__repr__() const {
  const auto nd = ndims();
  std::ostringstream oss;
  oss << "ddptensor{gs=(";
  for (auto i = 0; i < nd; ++i)
    oss << _gShape[i] << (i == nd - 1 ? "" : ", ");
  oss << "), loff=(";
  if (_lo_aligned)
    for (auto i = 0; i < nd; ++i)
      oss << _lo_aligned[i] << (i == nd - 1 ? "" : ", ");
  oss << "), lsz=(";
  for (auto i = 0; i < nd; ++i)
    oss << _lData._sizes[i] << (i == nd - 1 ? "" : ", ");
  oss << "), str=(";
  for (auto i = 0; i < nd; ++i)
    oss << _lData._strides[i] << (i == nd - 1 ? "" : ", ");
  oss << "), p=" << _lData._allocated << ", poff=" << _lData._offset
      << ", team=" << _transceiver << "}\n";

  dispatch(_dtype, _lData._aligned, [this, nd, &oss](auto *ptr) {
    auto cptr = ptr + this->_lData._offset;
    if (nd > 0) {
      printit(oss, 0, cptr);
    } else {
      oss << *cptr;
    }
  });
  return oss.str();
}

bool DDPTensorImpl::__bool__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar bool: tensor is not replicated"));

  bool res;
  dispatch(_dtype, _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<bool>(ptr[this->_lData._offset]);
  });
  return res;
}

double DDPTensorImpl::__float__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar float: tensor is not replicated"));

  double res;
  dispatch(_dtype, _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<double>(ptr[this->_lData._offset]);
  });
  return res;
}

int64_t DDPTensorImpl::__int__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar int: tensor is not replicated"));

  float res;
  dispatch(_dtype, _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<float>(ptr[this->_lData._offset]);
  });
  return res;
}

void DDPTensorImpl::add_to_args(std::vector<void *> &args) {
  int ndims = this->ndims();
  auto storeMR = [ndims](DynMemRef &mr) -> intptr_t * {
    intptr_t *buff = new intptr_t[dtensor_sz(ndims)];
    buff[0] = reinterpret_cast<intptr_t>(mr._allocated);
    buff[1] = reinterpret_cast<intptr_t>(mr._aligned);
    buff[2] = static_cast<intptr_t>(mr._offset);
    memcpy(buff + 3, mr._sizes, ndims * sizeof(intptr_t));
    memcpy(buff + 3 + ndims, mr._strides, ndims * sizeof(intptr_t));
    return buff;
  }; // FIXME memory leak?

  if (_transceiver == nullptr) {
    // no-dist-mode
    args.push_back(storeMR(_lData));
  } else {
    // transceiver/team first
    // args.push_back(_transceiver);
    // local tensor first
    if (ndims > 0) {
      args.push_back(storeMR(_lhsHalo));
      args.push_back(storeMR(_lData));
      args.push_back(storeMR(_rhsHalo));
      assert(5 == memref_sz(1));
      // local offsets last
      auto buff = new intptr_t[dtensor_sz(1)];
      buff[0] = reinterpret_cast<intptr_t>(_lo_allocated);
      buff[1] = reinterpret_cast<intptr_t>(_lo_aligned);
      buff[2] = 0;
      buff[3] = ndims;
      buff[4] = 1;
      args.push_back(buff);
    } else {
      args.push_back(storeMR(_lData));
    }
  }
}

void DDPTensorImpl::replicate() {
  if (is_replicated())
    return;
  auto gsz = size();
  auto lsz = local_size();
  if (gsz > 1)
    throw(std::runtime_error(
        "Replication implemented for single-element tensors only."));
  if (lsz != gsz) {
    assert(lsz == 0);
    auto nd = ndims();
    for (auto i = 0; i < nd; ++i) {
      _lData._sizes[i] = _lData._strides[i] = 1;
    }
    _lData._sizes[nd - 1] = gsz;
  }
  dispatch(_dtype, _lData._aligned, [this, lsz, gsz](auto *ptr) {
    auto tmp = ptr[this->_lData._offset];
    if (lsz != gsz)
      ptr[this->_lData._offset] = 0;
    getTransceiver()->reduce_all(&ptr[this->_lData._offset], this->_dtype, 1,
                                 SUM);
    assert(lsz != gsz || tmp == ptr[this->_lData._offset]);
  });
  set_owner(REPLICATED);
}
