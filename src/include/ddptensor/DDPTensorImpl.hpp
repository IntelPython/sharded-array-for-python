// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i.
// Interfaces are based on shared_ptr<tensor_i>.

#pragma once

#include "TypeDispatch.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <type_traits>

class Transceiver;

/// The actual implementation of the DDPTensor, implementing the tensor_i
/// interface. It holds the tensor data and some meta information. The member
/// attributes are mostly inspired by the needs of interacting with MLIR. It
/// also holds information needed for distributed operation.
class DDPTensorImpl : public tensor_i {
  mutable rank_type _owner;
  Transceiver *_transceiver = nullptr;
  void *_allocated = nullptr;
  void *_aligned = nullptr;
  intptr_t *_sizes = nullptr;
  intptr_t *_strides = nullptr;
  int64_t *_gs_allocated = nullptr;
  int64_t *_gs_aligned = nullptr;
  uint64_t *_lo_allocated = nullptr;
  uint64_t *_lo_aligned = nullptr;
  uint64_t _offset = 0;
  uint64_t _ndims = 0;
  uint64_t _balanced = 1;
  DTypeId _dtype = DTYPE_LAST;
  tensor_i::ptr_type _base;

public:
  using ptr_type = std::shared_ptr<DDPTensorImpl>;

  // don't allow copying.
  DDPTensorImpl(const DDPTensorImpl &) = delete;
  DDPTensorImpl(DDPTensorImpl &&) = default;

  // construct from a and MLIR-jitted execution
  DDPTensorImpl(Transceiver *transceiver, DTypeId dtype, uint64_t ndims,
                void *allocated, void *aligned, intptr_t offset,
                const intptr_t *sizes, const intptr_t *strides,
                int64_t *gs_allocated, int64_t *gs_aligned,
                uint64_t *lo_allocated, uint64_t *lo_aligned, uint64_t balanced,
                rank_type owner = NOOWNER);

  DDPTensorImpl(DTypeId dtype, const shape_type &shp,
                rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  DDPTensorImpl(const int64_t *shape, uint64_t N, rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  DDPTensorImpl() : _owner(REPLICATED) { assert(_ndims <= 1); }

  // @return a clone
  DDPTensorImpl::ptr_type clone(bool copy = true);

  // helper
  void alloc(bool all = true);

  // set the base tensor
  void set_base(const tensor_i::ptr_type &base) { _base = base; }

  virtual ~DDPTensorImpl();

  // @return pointer to raw data
  void *data();

  /// @return true if tensor is a sliced
  bool is_sliced() const {
    assert(!"Not implemented");
    return false;
  }

  /// python object's __repr__
  virtual std::string __repr__() const override;

  /// @return tensor's element type
  virtual DTypeId dtype() const override { return _dtype; }

  /// @return tensor's shape
  virtual const int64_t *shape() const override {
    return _transceiver ? _gs_aligned : local_shape();
  }

  /// @returnnumber of dimensions of tensor
  virtual int ndims() const override { return _ndims; }

  /// @return global number of elements in tensor
  virtual uint64_t size() const override {
    switch (ndims()) {
    case 0:
      return 1;
    case 1:
      return *_gs_aligned;
    default:
      return std::accumulate(_gs_aligned, _gs_aligned + ndims(), 1,
                             std::multiplies<intptr_t>());
    }
  }

  /// @return number of elements hold locally by calling process
  uint64_t local_size() const {
    switch (ndims()) {
    case 0:
      return 1;
    case 1:
      return *_sizes;
    default:
      return std::accumulate(_sizes, _sizes + ndims(), 1,
                             std::multiplies<intptr_t>());
    }
  }

  friend struct Service;

  /// @return boolean value of 0d tensor
  virtual bool __bool__() const override;
  /// @return float value of 0d tensor
  virtual double __float__() const override;
  /// @return integer value of 0d tensor
  virtual int64_t __int__() const override;

  /// @return global number of elements in first dimension
  virtual uint64_t __len__() const override {
    return ndims() ? *_gs_aligned : 1;
  }

  /// @return true if tensor has a unique owner
  bool has_owner() const { return _owner < _OWNER_END; }

  /// set the owner to given process rank
  void set_owner(rank_type o) const { _owner = o; }

  /// @return owning process rank or REPLICATED
  rank_type owner() const { return _owner; }

  /// @return Transceiver linked to this tensor
  Transceiver *transceiver() const { return _transceiver; }

  /// @return true if tensor's partitions are balanced
  uint64_t balanced() const { return _balanced; }

  /// @return true if tensor is replicated across all process ranks
  bool is_replicated() const { return _owner == REPLICATED; }

  /// @return size of one element in number of bytes
  virtual int item_size() const override { return sizeof_dtype(_dtype); }

  /// add tensor to list of args in the format expected by MLIR
  /// assuming tensor has ndims dims.
  virtual void add_to_args(std::vector<void *> &args) override;

  /// @return local offsets into global tensor
  const uint64_t *local_offsets() const { return _lo_aligned; }
  /// @return shape of local data
  const int64_t *local_shape() const { return _sizes; }
  /// @return strides of local data
  const int64_t *local_strides() const { return _strides; }

  // helper function for __repr__; simple recursive printing of
  // tensor content
  template <typename T>
  void printit(std::ostringstream &oss, uint64_t d, T *cptr) const {
    auto stride = _strides[d];
    auto sz = _sizes[d];
    if (d == ndims() - 1) {
      oss << "[";
      for (auto i = 0; i < sz; ++i) {
        oss << cptr[i * stride] << (i < sz - 1 ? " " : "");
      }
      oss << "]";
    } else {
      oss << "[";
      for (auto i = 0; i < sz; ++i) {
        if (i)
          oss << std::string(d + 1, ' ');
        printit(oss, d + 1, cptr);
        if (i < sz - 1)
          oss << "\n";
        cptr += stride;
      }
      oss << "]";
    }
  }

  void replicate();
};

/// create a new DDPTensor from given args and wrap in shared pointer
template <typename... Ts>
static typename DDPTensorImpl::ptr_type mk_tnsr(Ts &&...args) {
  return std::make_shared<DDPTensorImpl>(std::forward<Ts>(args)...);
}

template <typename... Ts> static tensor_i::future_type mk_ftx(Ts &&...args) {
  return UnDeferred(mk_tnsr(std::forward(args)...)).get_future();
}

// execute an OP on all elements of a tensor represented by
// dimensionality/ptr/sizes/strides.
template <typename T, typename OP, bool PASSIDX>
void forall_(uint64_t d, T *cptr, const int64_t *sizes, const int64_t *strides,
             uint64_t nd, OP op, std::vector<int64_t> *idx) {
  assert(!PASSIDX || idx);
  auto stride = strides[d];
  auto sz = sizes[d];
  if (d == nd - 1) {
    for (auto i = 0; i < sz; ++i) {
      if constexpr (PASSIDX) {
        (*idx)[d] = i;
        op(*idx, &cptr[i * stride]);
      } else if constexpr (!PASSIDX) {
        op(&cptr[i * stride]);
      }
    }
  } else {
    for (auto i = 0; i < sz; ++i) {
      T *tmp = cptr;
      if constexpr (PASSIDX) {
        (*idx)[d] = i;
      }
      forall_<T, OP, PASSIDX>(d + 1, cptr, sizes, strides, nd, op, idx);
      cptr = tmp + strides[d];
    }
  }
}

template <typename T, typename OP>
void forall(uint64_t d, T *cptr, const int64_t *sizes, const int64_t *strides,
            uint64_t nd, OP op) {
  forall_<T, OP, false>(d, cptr, sizes, strides, nd, op, nullptr);
}

template <typename T, typename OP>
void forall(uint64_t d, T *cptr, const int64_t *sizes, const int64_t *strides,
            uint64_t nd, std::vector<int64_t> &idx, OP op) {
  forall_<T, OP, true>(d, cptr, sizes, strides, nd, op, &idx);
}
