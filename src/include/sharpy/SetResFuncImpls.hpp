#include "CppTypes.hpp"

namespace SHARPY {
class Deferred;
struct DynMemRef;
void defaultSetResFunc(Deferred *dfrd, uint64_t rank, DynMemRef &&data,
                       DynMemRef &&splits, DynMemRef &&halos, DynMemRef &&offs,
                       id_type base);
} // namespace SHARPY
