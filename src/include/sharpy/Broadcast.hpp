
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "CppTypes.hpp"

namespace SHARPY {

template <typename V1, typename V2>
shape_type broadcast(const V1 &shape1, const V2 &shape2) {
  int64_t N1 = shape1.size();
  int64_t N2 = shape2.size();
  int64_t N = std::max(N1, N2);
  shape_type shape(N);

  for (int64_t i = N - 1; i >= 0; --i) {
    auto n1 = N1 - N + i;
    auto d1 = n1 >= 0 ? shape1[n1] : 1;
    auto n2 = N2 - N + i;
    auto d2 = n2 >= 0 ? shape2[n2] : 1;
    if (d1 == 1) {
      shape[i] = d2;
    } else if (d2 == 1) {
      shape[i] = d1;
    } else if (d1 == d2) {
      shape[i] = d1;
    } else {
      throw(std::runtime_error("Trying to broadcast incompatible shapes"));
    }
  }
  return shape;
}
} // namespace SHARPY
