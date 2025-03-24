// SPDX-License-Identifier: BSD-3-Clause

/*
  A registry of global arrays.
  Each array has a globally unique id.
*/

#include "sharpy/Registry.hpp"
#include <mutex>
#include <unordered_map>

namespace SHARPY {
namespace Registry {

using locker = std::lock_guard<std::mutex>;
using keeper_type =
    std::unordered_map<id_type, array_i::future_type>; //::weak_type>;
static keeper_type _keeper;
static std::mutex _mutex;
static id_type _nguid = -1;

id_type get_guid() { return ++_nguid; }

void put(const array_i::future_type &ptr) {
  locker _l(_mutex);
  _keeper.insert({ptr.guid(), ptr});
}

bool has(id_type id) {
  locker _l(_mutex);
  return _keeper.find(id) != _keeper.end();
}

array_i::future_type get(id_type id) {
  locker _l(_mutex);
  auto x = _keeper.find(id);
  if (x == _keeper.end())
    throw(std::runtime_error("Encountered request for unknown array."));
  return x->second; //.lock();
}

void del(id_type id) {
  locker _l(_mutex);
  _keeper.erase(id);
}

std::vector<id_type> get_all() {
  std::vector<id_type> res;
  locker _l(_mutex);
  for (auto f : _keeper) {
    res.emplace_back(f.first);
  }
  return res;
}

void fini() {
  locker _l(_mutex);
  _keeper.clear();
}
} // namespace Registry
} // namespace SHARPY
