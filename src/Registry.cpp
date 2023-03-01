#include "ddptensor/Registry.hpp"
#include <mutex>
#include <unordered_map>

namespace Registry {

using locker = std::lock_guard<std::mutex>;
using keeper_type =
    std::unordered_map<id_type, tensor_i::future_type>; //::weak_type>;
static keeper_type _keeper;
static std::mutex _mutex;
static id_type _nguid = -1;

id_type get_guid() { return ++_nguid; }

void put(const tensor_i::future_type &ptr) {
  locker _l(_mutex);
  _keeper[ptr.id()] = ptr;
}

tensor_i::future_type get(id_type id) {
  locker _l(_mutex);
  auto x = _keeper.find(id);
  if (x == _keeper.end())
    throw(std::runtime_error("Encountered request for unknown tensor."));
  return x->second; //.lock();
}

void del(id_type id) {
  locker _l(_mutex);
  _keeper.erase(id);
}

void fini() {
  locker _l(_mutex);
  _keeper.clear();
}
} // namespace Registry
