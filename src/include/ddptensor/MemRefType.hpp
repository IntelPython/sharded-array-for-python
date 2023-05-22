// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Generic descriptor of rank N strided MemRef
template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated = nullptr;
  T *aligned = nullptr;
  intptr_t offset = 0;
  intptr_t sizes[N] = {0};
  intptr_t strides[N] = {0};
};

// Use with care.
template <typename T> class UnrankedMemRefType {
  int64_t _rank;
  intptr_t *_descriptor;

public:
  UnrankedMemRefType(int64_t rank, void *p)
      : _rank(rank), _descriptor(reinterpret_cast<intptr_t *>(p)){};

  T *data() { return reinterpret_cast<T *>(_descriptor[1]); };
  int64_t rank() const { return _rank; }
  int64_t *sizes() { return reinterpret_cast<int64_t *>(&_descriptor[3]); };
  int64_t *strides() {
    return reinterpret_cast<int64_t *>(&_descriptor[3 + _rank]);
  };
};

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
  T *data() { return descriptor->aligned; };
  int64_t size() { return descriptor->sizes[0]; };
};
