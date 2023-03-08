// SPDX-License-Identifier: BSD-3-Clause

/*
  I/O ops.
*/

#include "ddptensor/IO.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/Transceiver.hpp"
#include "ddptensor/TypeDispatch.hpp"

GetItem::py_future_type IO::to_numpy(const ddptensor &a) {
  assert(!getTransceiver()->is_cw() || getTransceiver()->rank() == 0);
  return GetItem::gather(a, 0);
}
