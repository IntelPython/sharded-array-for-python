// SPDX-License-Identifier: BSD-3-Clause

// the queue of deferred runables

#include "include/sharpy/Deferred.hpp"
#include <oneapi/tbb/concurrent_queue.h>

namespace SHARPY {

// thread-safe FIFO queue holding deferred objects
tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

// add a deferred object to the queue
void push_runable(Runable::ptr_type &&r) { _deferred.push(std::move(r)); }

} // namespace SHARPY
