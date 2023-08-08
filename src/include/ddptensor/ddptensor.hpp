// SPDX-License-Identifier: BSD-3-Clause

/*
    The tensor class exposed to Python.
    A future type, data and implementation live elsewhere
    (see tensor_i, DDPTensorImpl and Deferred).
*/

#pragma once

#include "Service.hpp"
#include "tensor_i.hpp"

class ddptensor {
  tensor_i::future_type _ftx;

public:
  ddptensor(tensor_i::future_type &&f)
      : _ftx(std::forward<tensor_i::future_type>(f)) {}
  ddptensor(std::shared_future<tensor_i::ptr_type> &&f, id_type id, DTypeId dt,
            const shape_type &shape, uint64_t team, bool balanced)
      : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt,
             shape, team, balanced) {}
  ddptensor(std::shared_future<tensor_i::ptr_type> &&f, id_type id, DTypeId dt,
            shape_type &&shape, uint64_t team, bool balanced)
      : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt,
             std::forward<shape_type>(shape), team, balanced) {}

  ~ddptensor() { Service::drop(*this); }

  const tensor_i::future_type &get() const { return _ftx; }
  void const put(tensor_i::future_type &&f) {
    _ftx = std::forward<tensor_i::future_type>(f);
  }
};
