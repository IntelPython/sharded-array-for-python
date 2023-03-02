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
  ddptensor(const tensor_i::future_type &f) : _ftx(f) {}
  ddptensor(std::shared_future<tensor_i::ptr_type> &&f, id_type id, DTypeId dt,
            int rank, bool balanced)
      : _ftx(std::forward<std::shared_future<tensor_i::ptr_type>>(f), id, dt,
             rank, balanced) {}

  ~ddptensor() { Service::drop(*this); }

  const tensor_i::future_type &get() const { return _ftx; }
};
