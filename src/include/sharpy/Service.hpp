// SPDX-License-Identifier: BSD-3-Clause

/*
 Service operations, mostly used internally.
*/

#pragma once

#include "CppTypes.hpp"
#include <future>

namespace SHARPY {

class FutureArray;
class NDArray;

struct Service {
  using service_promise_type = std::promise<bool>;
  using service_future_type = std::shared_future<bool>;

  /// replicate the given FutureArray on all ranks
  static FutureArray *replicate(const FutureArray &a);
  /// start running/executing operations, e.g. trigger compile&run
  /// this is not blocking, use futures for synchronization
  static service_future_type run();
  /// signal that the given FutureArray is no longer needed and can be deleted
  static void drop(id_type a);
};
} // namespace SHARPY
