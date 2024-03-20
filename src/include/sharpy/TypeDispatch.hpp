// SPDX-License-Identifier: BSD-3-Clause

/*
  Helpers for dispatching computation dependent on dynamic types (DTypeId).
*/

#pragma once

#include "CppTypes.hpp"

namespace SHARPY {
// Dependent on dt, dispatch arguments to a operation class.
// The operation must
//    * be a template class accepting the element type as argument
//    * implement one or more "op" methods matching the given arguments (args)
// All arguments other than dt are opaquely passed to the operation.
template <template <typename OD> class OpDispatch, typename... Ts>
auto dispatch(DTypeId dt, Ts &&...args) {
  switch (dt) {
  case FLOAT64:
    return OpDispatch<double>::op(std::forward<Ts>(args)...);
  case INT64:
    return OpDispatch<int64_t>::op(std::forward<Ts>(args)...);
  case FLOAT32:
    return OpDispatch<float>::op(std::forward<Ts>(args)...);
  case INT32:
    return OpDispatch<int32_t>::op(std::forward<Ts>(args)...);
  case INT16:
    return OpDispatch<int16_t>::op(std::forward<Ts>(args)...);
  case INT8:
    return OpDispatch<int8_t>::op(std::forward<Ts>(args)...);
  case UINT64:
    return OpDispatch<uint64_t>::op(std::forward<Ts>(args)...);
  case UINT32:
    return OpDispatch<uint32_t>::op(std::forward<Ts>(args)...);
  case UINT16:
    return OpDispatch<uint16_t>::op(std::forward<Ts>(args)...);
  case UINT8:
    return OpDispatch<uint8_t>::op(std::forward<Ts>(args)...);
  case BOOL:
    return OpDispatch<bool>::op(std::forward<Ts>(args)...);
  default:
    throw std::invalid_argument("unknown dtype");
  }
}

// dispatch template for simple functions
// for example for lambdas accepting the pointer as auto
template <typename T> struct funcDispatcher {
  template <typename DispatchFunc>
  static void op(void *ptr, DispatchFunc func) {
    func(reinterpret_cast<T *>(ptr));
  }
  template <typename DispatchFunc>
  static void op(const void *ptr, DispatchFunc func) {
    func(reinterpret_cast<const T *>(ptr));
  }
};

// shortcut for simple function/lambda dispatch
template <typename DispatchFunc>
void dispatch(DTypeId dt, void *ptr, DispatchFunc func) {
  dispatch<funcDispatcher>(dt, ptr, func);
}
template <typename DispatchFunc>
void dispatch(DTypeId dt, const void *ptr, DispatchFunc func) {
  dispatch<funcDispatcher>(dt, ptr, func);
}

} // namespace SHARPY
