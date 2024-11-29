// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of array_i.
// Interfaces are based on shared_ptr<array_i>.

#pragma once

#include "MemRefType.hpp"
#include "Registry.hpp"
#include "Transceiver.hpp"
#include "TypeDispatch.hpp"
#include "array_i.hpp"
#include "p2c_ids.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <type_traits>

namespace SHARPY {
class Transceiver;

/// The actual implementation of the SHARPYensor, implementing the array_i
/// interface. It holds the array data and some meta information. The member
/// attributes are mostly inspired by the needs of interacting with MLIR. It
/// also holds information needed for distributed operation.
/// Besides the local data it also holds data haloed from other processes.
/// Here, the halos are never used for anything except for interchanging with
/// MLIR.
class NDArray : public array_i, protected ArrayMeta {
  mutable rank_type _owner = NOOWNER;
  DynMemRef _lhsHalo;
  DynMemRef _lData;
  DynMemRef _rhsHalo;
  std::vector<int64_t> _lOffsets;
  BaseObj *_base = nullptr;

public:
  struct NDADeleter {
    void operator()(NDArray *a) const;
  };
  friend struct NDADeleter;

  using ptr_type = std::shared_ptr<NDArray>;

  // don't allow copying.
  NDArray(const NDArray &) = delete;
  NDArray(NDArray &&) = default;

  // construct from a and MLIR-jitted execution
  NDArray(id_type guid, DTypeId dtype, shape_type gShape,
          const std::string &device, const std::string &team, void *l_allocated,
          void *l_aligned, intptr_t l_offset, const intptr_t *l_sizes,
          const intptr_t *l_strides, void *o_allocated, void *o_aligned,
          intptr_t o_offset, const intptr_t *o_sizes, const intptr_t *o_strides,
          void *r_allocated, void *r_aligned, intptr_t r_offset,
          const intptr_t *r_sizes, const intptr_t *r_strides,
          std::vector<int64_t> &&loffs, rank_type owner = NOOWNER);

  NDArray(id_type guid, DTypeId dtype, const shape_type &shp,
          const std::string &device, const std::string &team,
          rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  NDArray(id_type guid, const int64_t *shape, uint64_t N,
          const std::string &device, const std::string &team,
          rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  NDArray() : _owner(REPLICATED) {
    if (ndims() > 1)
      throw std::runtime_error("Incorrect NDArray construction.");
  }

  // From numpy
  // FIXME multi-proc
  NDArray(id_type guid, DTypeId dtype, ssize_t ndims, const ssize_t *shape,
          const intptr_t *strides, void *data, const std::string &device,
          const std::string &team);

  // set the base array
  void set_base(const array_i::ptr_type &base);
  void set_base(BaseObj *obj);

  virtual ~NDArray();

  // mark local data and halos as deallocated
  void markDeallocated() override;
  bool isAllocated() override;

  // @return pointer to raw data
  void *data();

  /// @return true if array is a sliced
  bool is_sliced() const;

  /// python object's __repr__
  virtual std::string __repr__() const override;

  /// @return array's GUID
  virtual id_type guid() const { return ArrayMeta::guid(); }

  /// @return array's element type
  virtual DTypeId dtype() const override { return ArrayMeta::dtype(); }

  /// @return array's shape
  virtual const shape_type &shape() const override {
    return ArrayMeta::shape();
  }

  /// @returnnumber of dimensions of array
  virtual int ndims() const override { return ArrayMeta::rank(); }

  const std::string &team() const { return ArrayMeta::team(); }
  const std::string &device() const { return ArrayMeta::device(); }

  /// @return global number of elements in array
  virtual uint64_t size() const override {
    switch (ndims()) {
    case 0:
      return 1;
    case 1:
      return ArrayMeta::shape().front();
    default:
      return std::accumulate(ArrayMeta::shape().begin(),
                             ArrayMeta::shape().end(), 1,
                             std::multiplies<intptr_t>());
    }
  }

  /// @return number of elements hold locally by calling process
  uint64_t local_size() const {
    switch (ndims()) {
    case 0:
      return 1;
    case 1:
      return *_lData._sizes;
    default:
      return std::accumulate(_lData._sizes, _lData._sizes + ndims(), 1,
                             std::multiplies<intptr_t>());
    }
  }

  friend struct Service;

  /// @return boolean value of 0d array
  virtual bool __bool__() const override;
  /// @return float value of 0d array
  virtual double __float__() const override;
  /// @return integer value of 0d array
  virtual int64_t __int__() const override;

  /// @return global number of elements in first dimension
  virtual uint64_t __len__() const override {
    return ndims() ? ArrayMeta::shape().front() : 1;
  }

  /// @return true if array has a unique owner
  bool has_owner() const { return _owner < _OWNER_END; }

  /// set the owner to given process rank
  void set_owner(rank_type o) const { _owner = o; }

  /// @return owning process rank or REPLICATED
  rank_type owner() const { return _owner; }

  /// @return Transceiver linked to this array
  // FIXME
  Transceiver *transceiver() const { return getTransceiver(); }

  /// @return true if array is replicated across all process ranks
  bool is_replicated() const { return _owner == REPLICATED; }

  /// @return size of one element in number of bytes
  virtual int item_size() const override { return sizeof_dtype(_dtype); }

  /// @return locally owned data as DynMemref
  const DynMemRef &owned_data() const { return _lData; }
  /// @return left halo as DynMemref
  const DynMemRef &left_halo() const { return _lhsHalo; }
  /// @return right halo as DynMemref
  const DynMemRef &right_halo() const { return _rhsHalo; }

  /// @return local offsets into global array
  const std::vector<int64_t> &local_offsets() const { return _lOffsets; }
  /// @return shape of local data
  const int64_t *local_shape() const { return _lData._sizes; }
  /// @return strides of local data
  const int64_t *local_strides() const { return _lData._strides; }
  /// @return shape of left halo
  const int64_t *lh_shape() const { return _lhsHalo._sizes; }
  /// @return shape of right halo
  const int64_t *rh_shape() const { return _rhsHalo._sizes; }

  // helper function for __repr__; simple recursive printing of
  // array content
  template <typename T>
  void printit(std::ostringstream &oss, uint64_t d, T *cptr) const {
    auto stride = _lData._strides[d];
    auto sz = _lData._sizes[d];
    if (d == (uint64_t)ndims() - 1) {
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

/// create a new SHARPYensor from given args and wrap in shared pointer
template <typename... Ts>
static typename NDArray::ptr_type mk_tnsr(Ts &&...args) {
  return NDArray::ptr_type(new NDArray(std::forward<Ts>(args)...),
                           NDArray::NDADeleter());
}

// execute an OP on all elements of a array represented by
// dimensionality/ptr/sizes/strides.
template <typename T, typename OP, bool PASSIDX>
void forall_(uint64_t d, T *cptr, const int64_t *sizes, const int64_t *strides,
             uint64_t nd, OP op, std::vector<int64_t> *idx) {
  if (PASSIDX && !idx) {
    throw std::invalid_argument(
        "Internal error: cannot perform forall on nullptr.");
  }
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
} // namespace SHARPY
