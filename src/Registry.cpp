#include <unordered_map>
#include <mutex>
#include "ddptensor/Registry.hpp"

namespace Registry {

    using locker = std::lock_guard<std::mutex>;
    using keeper_type = std::unordered_map<id_type, tensor_i::ptr_type>; //::weak_type>;
    static keeper_type _keeper;
    static std::mutex _mutex;
    static id_type _nguid = -1;

    id_type get_guid()
    {
        return ++_nguid;
    }

    void put(id_type id, tensor_i::ptr_type ptr)
    {
        // std::cerr << "Registry::put(" << id << ")\n";
        locker _l(_mutex);
        _keeper[id] = ptr;
    }
    
    tensor_i::ptr_type get(id_type id)
    {
        // std::cerr << "Registry::get(" << id << ")\n";
        locker _l(_mutex);
        auto x = _keeper.find(id);
        if(x == _keeper.end()) throw(std::runtime_error("Encountered request for unknown tensor."));
        return x->second; //.lock();
    }

    void del(id_type id)
    {
        // std::cerr << "Registry::del(" << id << ")\n";
        locker _l(_mutex);
        _keeper.erase(id);
    }
}
