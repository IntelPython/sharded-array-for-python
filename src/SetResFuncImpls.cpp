#include <sharpy/Deferred.hpp>
#include <sharpy/NDArray.hpp>
#include <sharpy/Registry.hpp>
#include <sharpy/SetResFuncImpls.hpp>

namespace SHARPY {

void defaultSetResFunc(Deferred *dfrd, uint64_t rank, DynMemRef &&data,
                       DynMemRef &&splits, DynMemRef &&halos, DynMemRef &&offs,
                       id_type base) {
  auto t = mk_tnsr(dfrd->guid(), dfrd->dtype(), dfrd->shape(), dfrd->device(),
                   dfrd->team(), std::move(data), std::move(splits),
                   std::move(halos), std::move(offs));
  if (base != NOGUID) {
    t->set_base(Registry::get(base).get());
  }
  dfrd->set_value(std::move(t));
}

} // namespace SHARPY
