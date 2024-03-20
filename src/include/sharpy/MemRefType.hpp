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
  int64_t _rank = -1;
  intptr_t *_descriptor = nullptr;

public:
  void validate() const {
    if (_rank < 0) {
      throw std::out_of_range("Invalid rank<0 in UnrankedMemRefType.");
    }
    if (!_descriptor) {
      throw std::invalid_argument(
          "Invalid descriptor==nullptr in UnrankedMemRefType.");
    }
    auto allocated = _descriptor[0];
    auto aligned = _descriptor[1];
    auto off = _descriptor[2];
    if (allocated == 0 || aligned == 0) {
      throw std::invalid_argument("Invalid nullptr in UnrankedMemRefType.");
    }
    if (off < 0) {
      throw std::out_of_range("Invalid offset<0 in UnrankedMemRefType.");
    }
    auto start = aligned + off;
    if (start < aligned) {
      throw std::overflow_error(
          "Fatal: Integer overflow in UnrankedMemRefType.");
    }
    if (_rank) {
      auto szs = sizes();
      auto strs = strides();
      for (auto i = 0; i < _rank; ++i) {
        auto sz = szs[i];
        if (sz < 0) {
          throw std::out_of_range("Invalid size<0 in UnrankedMemRefType.");
        } else if (sz > 0) {
          auto str = strs[i];
          // FIXME negative strides
          if (str <= 0) {
            throw std::out_of_range("Invalid stride<0 in UnrankedMemRefType.");
          }
          auto x = sz * str;
          if (x / sz != str) {
            throw std::overflow_error(
                "Fatal: Integer overflow in UnrankedMemRefType.");
          }
          if (start + x < start) {
            throw std::overflow_error(
                "Fatal: Integer overflow in UnrankedMemRefType.");
          }
        }
      }
    }
  }

  UnrankedMemRefType(int64_t rank, void *p)
      : _rank(rank), _descriptor(reinterpret_cast<intptr_t *>(p)) {
    validate();
  };

  T *data() { return &reinterpret_cast<T *>(_descriptor[1])[_descriptor[2]]; };
  int64_t rank() const { return _rank; }
  int64_t *sizes() {
    return _rank ? reinterpret_cast<int64_t *>(&_descriptor[3]) : nullptr;
  };
  const int64_t *sizes() const {
    return const_cast<UnrankedMemRefType<T> *>(this)->sizes();
  };
  int64_t *strides() {
    return _rank ? reinterpret_cast<int64_t *>(&_descriptor[3 + _rank])
                 : nullptr;
  };
  const int64_t *strides() const {
    return const_cast<UnrankedMemRefType<T> *>(this)->strides();
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
    if (idx >= d->sizes[0] || idx < 0) {
      throw std::out_of_range("Index out of bounds in Unranked1DMemRefType.");
    }
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

  void validate(uint64_t ndims) const {
    if (!((_allocated && _aligned && (ndims == 0 || (_sizes && _strides))) ||
          (!_allocated && !_aligned && !_sizes && !_strides && ndims == 0))) {
      throw std::invalid_argument("Invalid nullptr in DynMemRef.");
    }
    for (auto i = 0u; i < ndims; ++i) {
      if (_sizes[i] < 0) {
        throw std::out_of_range("Invalid size<0 in DynMemRef.");
      }
      if (_sizes[i]) {
        if (_strides[i]) {
          auto x = _strides[i] * i;
          if (x / _strides[i] != i || _sizes[i] + x < _sizes[i]) {
            throw std::overflow_error("Fatal: Integer overflow in DynMemRef.");
          }
        } else {
          throw std::out_of_range("Invalid stride==0 in DynMemRef.");
        }
      }
    }
  }

  DynMemRef(uint64_t ndims, void *allocated, void *aligned, intptr_t offset,
            const intptr_t *sizes, const intptr_t *strides)
      : _offset(offset), _allocated(allocated), _aligned(aligned) {
    if (ndims > 0) {
      _sizes = new intptr_t[ndims];
      _strides = new intptr_t[ndims];
      memcpy(_sizes, sizes, ndims * sizeof(*_sizes));
      memcpy(_strides, strides, ndims * sizeof(*_strides));
    }
    validate(ndims);
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
