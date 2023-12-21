// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace SHARPY {

/// @return true if size/strides represent a contiguous data layout
inline bool is_contiguous(const int64_t *sizes, const int64_t *strides,
                          uint64_t ndims) {
  if (ndims == 0)
    return true;
  if (strides[ndims - 1] != 1)
    return false;
  auto sz = 1;
  for (auto i = ndims - 1; i > 0; --i) {
    sz *= sizes[i];
    if (strides[i - 1] != sz)
      return false;
  }
  return true;
}

/// @brief Generic descriptor of rank N strided MemRef
template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated = nullptr;
  T *aligned = nullptr;
  intptr_t offset = 0;
  intptr_t sizes[N] = {0};
  intptr_t strides[N] = {0};
};

/// @brief Generic dtype templated MemRef type
/// Provides access to data, sizes, and strides pointers
template <typename T> class UnrankedMemRefType {
  int64_t _rank;
  intptr_t *_descriptor;

public:
  UnrankedMemRefType(int64_t rank, void *p)
      : _rank(rank), _descriptor(reinterpret_cast<intptr_t *>(p)){};

  T *data() { return &reinterpret_cast<T *>(_descriptor[1])[_descriptor[2]]; };
  int64_t rank() const { return _rank; }
  int64_t *sizes() {
    return _rank ? reinterpret_cast<int64_t *>(&_descriptor[3]) : nullptr;
  };
  int64_t *strides() {
    return _rank ? reinterpret_cast<int64_t *>(&_descriptor[3 + _rank])
                 : nullptr;
  };
  bool contiguous_layout() {
    return is_contiguous(sizes(), strides(), rank());
  };
};

/// @brief 1D dtype templated MemRef type
/// Provides [] operator access to data
template <typename T> struct Unranked1DMemRefType {
  MemRefDescriptor<T, 1> *descriptor;

  Unranked1DMemRefType(int64_t rank, void *p)
      : descriptor(static_cast<MemRefDescriptor<T, 1> *>(p)) {
    assert(rank == 1);
  };

  T &operator[](int64_t idx) {
    auto d = descriptor;
    return *(d->aligned + d->offset + idx * d->strides[0]);
  };
  T *data() { return descriptor->aligned + descriptor->offset; };
  int64_t size() { return descriptor->sizes[0]; };
};

// This struct keeps all data generically for interchanging arrays/memrefs with
// LLVM/MLIR rank-dependent such as strides ans shape data gets copied into own
// memory
struct DynMemRef {
  intptr_t _offset = 0;
  void *_allocated = nullptr;
  void *_aligned = nullptr;
  intptr_t *_sizes = nullptr;
  intptr_t *_strides = nullptr;

  DynMemRef(uint64_t ndims, void *allocated, void *aligned, intptr_t offset,
            const intptr_t *sizes, const intptr_t *strides)
      : _offset(offset), _allocated(allocated), _aligned(aligned) {
    if (ndims > 0) {
      _sizes = new intptr_t[ndims];
      _strides = new intptr_t[ndims];
      memcpy(_sizes, sizes, ndims * sizeof(*_sizes));
      memcpy(_strides, strides, ndims * sizeof(*_strides));
    }
  }

  DynMemRef(const DynMemRef &) = delete;
  DynMemRef() = default;
  DynMemRef(DynMemRef &&src)
      : _offset(src._offset), _allocated(src._allocated),
        _aligned(src._aligned), _sizes(src._sizes), _strides(src._strides) {
    src._sizes = src._strides = nullptr;
    src._allocated = src._aligned = nullptr;
    src._offset = 0;
  }

  DynMemRef &operator=(const DynMemRef &src) = delete;
  DynMemRef &operator=(DynMemRef &&src) {
    _offset = src._offset;
    _allocated = src._allocated;
    _aligned = src._aligned;
    _sizes = src._sizes;
    _strides = src._strides;
    src._sizes = src._strides = nullptr;
    src._allocated = src._aligned = nullptr;
    src._offset = 0;
    return *this;
  }

  ~DynMemRef() {
    delete[] _sizes;
    delete[] _strides;
  }

  void freeData() {
    if (_allocated) {
      free(_allocated);
      markDeallocated();
    }
  }

  void markDeallocated() { _allocated = nullptr; }
};
} // namespace SHARPY
