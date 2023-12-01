// SPDX-License-Identifier: BSD-3-Clause

/*
    The tensor class exposed to Python.
    A future type, data and implementation live elsewhere
    (see tensor_i, DDPTensorImpl and Deferred).
*/

#pragma once

#include "Service.hpp"
#include "tensor_i.hpp"

namespace DDPT {

class ddptensor {
  tensor_i::future_type _ftx;

public:
  ddptensor(tensor_i::future_type &&f)
      : _ftx(std::forward<tensor_i::future_type>(f)) {}
  ddptensor(std::shared_future<tensor_i::ptr_type> &&f, id_type id, DTypeId dt,
            const shape_type &shape, const std::string &device, uint64_t team)
      : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt,
             shape, device, team) {}
  ddptensor(std::shared_future<tensor_i::ptr_type> &&f, id_type id, DTypeId dt,
            shape_type &&shape, std::string &&device, uint64_t team)
      : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt,
             std::forward<shape_type>(shape), std::forward<std::string>(device),
             team) {}

  ~ddptensor() { Service::drop(*this); }

  const tensor_i::future_type &get() const { return _ftx; }
  void const put(tensor_i::future_type &&f) {
    _ftx = std::forward<tensor_i::future_type>(f);
  }
};
} // namespace DDPT
