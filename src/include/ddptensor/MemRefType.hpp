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

template <typename T> struct Unranked1DMemRefType {
  int64_t rank;
  MemRefDescriptor<T, 1> *descriptor;

  Unranked1DMemRefType(int64_t _rank, void *p)
      : rank(_rank), descriptor(static_cast<MemRefDescriptor<T, 1> *>(p)) {
    assert(rank == 1);
  };

  T &operator[](int64_t idx) {
    auto d = descriptor;
    return *(d->aligned + d->offset + idx * d->strides[0]);
  };
  T *data() { return descriptor->aligned; };
};
