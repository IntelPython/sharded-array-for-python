// SPDX-License-Identifier: BSD-3-Clause

/*
  A registry of global tensors.
  Each tensor has a globally unique id.
*/

#pragma once

#include "tensor_i.hpp"

namespace Registry {

constexpr static id_type NOGUID = -1;

/// @return a new (unused) guid
id_type get_guid();

/// store a future tensor to registry
/// assumes guid has already been assigned
extern void put(const tensor_i::future_type &ptr);

/// @return future tensor with guid id
tensor_i::future_type get(id_type id);

/// remove future tensor with guid id from registry
void del(id_type id);

/// @return true if given guid is registered
bool has(id_type);

/// finalize registry (before shutdown)
void fini();

}; // namespace Registry
