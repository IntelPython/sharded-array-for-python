// SPDX-License-Identifier: BSD-3-Clause

/*
  A registry of global arrays.
  Each array has a globally unique id.
*/

#pragma once

#include "array_i.hpp"

namespace SHARPY {
namespace Registry {

constexpr static id_type NOGUID = -1;

/// @return a new (unused) guid
id_type get_guid();

/// store a future array to registry
/// assumes guid has already been assigned
extern void put(const array_i::future_type &ptr);

/// @return future array with guid id
array_i::future_type get(id_type id);

/// remove future array with guid id from registry
void del(id_type id);

/// @return true if given guid is registered
bool has(id_type);

/// @return all (currently) registered arrays in vector
std::vector<id_type> get_all();

/// finalize registry (before shutdown)
void fini();

}; // namespace Registry
} // namespace SHARPY
