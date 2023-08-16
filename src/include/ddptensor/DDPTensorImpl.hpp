// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of tensor_i.
// Interfaces are based on shared_ptr<tensor_i>.

#pragma once

#include "MemRefType.hpp"
#include "TypeDispatch.hpp"
#include "p2c_ids.hpp"
#include "tensor_i.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <type_traits>

class Transceiver;

/// @brief use this to provide a base object to the tensor
// such a base object can own shared data
// you might need to implem,ent reference counting
struct BaseObj {
  virtual ~BaseObj() {}
};

/// @brief Simple implementatino of BaseObj for ref-counting types
/// @tparam T ref-counting type, such as py::object of std::shared_Ptr
/// we keep an object of the ref-counting type. Normal ref-counting/destructors
/// will take care of the rest.
template <typename T> struct SharedBaseObject : public BaseObj {
  SharedBaseObject(const SharedBaseObject &) = default;
  SharedBaseObject(SharedBaseObject &&) = default;
  SharedBaseObject(const T &o) : _base(o) {}
  SharedBaseObject(T &&o) : _base(std::forward<T>(o)) {}
  T _base;
};

/// The actual implementation of the DDPTensor, implementing the tensor_i
/// interface. It holds the tensor data and some meta information. The member
/// attributes are mostly inspired by the needs of interacting with MLIR. It
/// also holds information needed for distributed operation.
/// Besides the local data it alos holds data haloed from other processes.
/// Here, the halos are never used for anything except for interchanging with
/// MLIR.
class DDPTensorImpl : public tensor_i {

  mutable rank_type _owner;
  Transceiver *_transceiver = nullptr;
  shape_type _gShape = {};
  uint64_t *_lo_allocated = nullptr;
  uint64_t *_lo_aligned = nullptr;
  DynMemRef _lhsHalo;
  DynMemRef _lData;
  DynMemRef _rhsHalo;
  DTypeId _dtype = DTYPE_LAST;
  BaseObj *_base = nullptr;

public:
  using ptr_type = std::shared_ptr<DDPTensorImpl>;

  // don't allow copying.
  DDPTensorImpl(const DDPTensorImpl &) = delete;
  DDPTensorImpl(DDPTensorImpl &&) = default;

  // construct from a and MLIR-jitted execution
  DDPTensorImpl(Transceiver *transceiver, DTypeId dtype, shape_type gShape,
                void *l_allocated, void *l_aligned, intptr_t l_offset,
                const intptr_t *l_sizes, const intptr_t *l_strides,
                void *o_allocated, void *o_aligned, intptr_t o_offset,
                const intptr_t *o_sizes, const intptr_t *o_strides,
                void *r_allocated, void *r_aligned, intptr_t r_offset,
                const intptr_t *r_sizes, const intptr_t *r_strides,
                uint64_t *lo_allocated, uint64_t *lo_aligned,
                rank_type owner = NOOWNER);

  DDPTensorImpl(DTypeId dtype, const shape_type &shp,
                rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  DDPTensorImpl(const int64_t *shape, uint64_t N, rank_type owner = NOOWNER);

  // incomplete, useful for computing meta information
  DDPTensorImpl() : _owner(REPLICATED) { assert(ndims() <= 1); }

  // From numpy
  // FIXME multi-proc
  DDPTensorImpl(DTypeId dtype, ssize_t ndims, const ssize_t *shape,
                const intptr_t *strides, void *data);

  // set the base tensor
  void set_base(const tensor_i::ptr_type &base);
  void set_base(BaseObj *obj);

  virtual ~DDPTensorImpl();

  // @return pointer to raw data
  void *data();

  /// @return true if tensor is a sliced
  bool is_sliced() const;

  /// python object's __repr__
  virtual std::string __repr__() const override;

  /// @return tensor's element type
  virtual DTypeId dtype() const override { return _dtype; }

  /// @return tensor's shape
  virtual const int64_t *shape() const override {
    return _transceiver ? _gShape.data() : local_shape();
  }

  /// @returnnumber of dimensions of tensor
  virtual int ndims() const override { return _gShape.size(); }

  /// @return global number of elements in tensor
  virtual uint64_t size() const override {
    switch (ndims()) {
    case 0:
      return 1;
    case 1:
      return _gShape.front();
    default:
      return std::accumulate(_gShape.begin(), _gShape.end(), 1,
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

  /// @return boolean value of 0d tensor
  virtual bool __bool__() const override;
  /// @return float value of 0d tensor
  virtual double __float__() const override;
  /// @return integer value of 0d tensor
  virtual int64_t __int__() const override;

  /// @return global number of elements in first dimension
  virtual uint64_t __len__() const override {
    return ndims() ? _gShape.front() : 1;
  }

  /// @return true if tensor has a unique owner
  bool has_owner() const { return _owner < _OWNER_END; }

  /// set the owner to given process rank
  void set_owner(rank_type o) const { _owner = o; }

  /// @return owning process rank or REPLICATED
  rank_type owner() const { return _owner; }

  /// @return Transceiver linked to this tensor
  Transceiver *transceiver() const { return _transceiver; }

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
  const int64_t *local_shape() const { return _lData._sizes; }
  /// @return strides of local data
  const int64_t *local_strides() const { return _lData._strides; }
  /// @return shape of left halo
  const int64_t *lh_shape() const { return _lhsHalo._sizes; }
  /// @return shape of right halo
  const int64_t *rh_shape() const { return _rhsHalo._sizes; }

  // helper function for __repr__; simple recursive printing of
  // tensor content
  template <typename T>
  void printit(std::ostringstream &oss, uint64_t d, T *cptr) const {
    auto stride = _lData._strides[d];
    auto sz = _lData._sizes[d];
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
