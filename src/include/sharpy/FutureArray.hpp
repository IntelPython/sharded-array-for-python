// SPDX-License-Identifier: BSD-3-Clause

/*
    The array class exposed to Python.
    A future type, data and implementation live elsewhere
    (see array_i, NDArray and Deferred).
*/

#pragma once

#include "Service.hpp"
#include "array_i.hpp"

namespace SHARPY {

extern bool finied;

class FutureArray {
  array_i::future_type _ftx;

public:
  FutureArray(array_i::future_type &&f)
      : _ftx(std::forward<array_i::future_type>(f)) {}
  FutureArray(std::shared_future<array_i::ptr_type> &&f, id_type id, DTypeId dt,
              const shape_type &shape, const std::string &device, uint64_t team)
      : _ftx(std::forward<std::shared_future<array_i::ptr_type>>(f), id, dt,
             shape, device, team) {}
  FutureArray(std::shared_future<array_i::ptr_type> &&f, id_type id, DTypeId dt,
              shape_type &&shape, std::string &&device, uint64_t team)
      : _ftx(std::forward<std::shared_future<array_i::ptr_type>>(f), id, dt,
             std::forward<shape_type>(shape), std::forward<std::string>(device),
             team) {}

  ~FutureArray() {
    if (!finied)
      Service::drop(get().guid());
  }

  const array_i::future_type &get() const { return _ftx; }
  void put(array_i::future_type &&f) {
    _ftx = std::forward<array_i::future_type>(f);
  }
};
} // namespace SHARPY
