// SPDX-License-Identifier: BSD-3-Clause

/*
    C++ types and utils. No Python/Pybind11 dependences.
*/

#pragma once

#include "p2c_ids.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include <bitsery/adapter/buffer.h>
#include <bitsery/bitsery.h>
#include <bitsery/traits/vector.h>

namespace SHARPY {

using shape_type = std::vector<int64_t>;
using dim_vec_type = std::vector<int>;
using rank_type = uint64_t;
using Buffer = std::vector<uint8_t>;
using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;
using InputAdapter = bitsery::InputBufferAdapter<Buffer>;
using Serializer = bitsery::Serializer<OutputAdapter>;
using Deserializer = bitsery::Deserializer<InputAdapter>;

union PyScalar {
  int64_t _int;
  double _float;
};

inline bool is_none(PyScalar s) { return std::isnan(s._float); }

enum _RANKS : rank_type {
  NOOWNER = std::numeric_limits<rank_type>::max(),
  REPLICATED = std::numeric_limits<rank_type>::max() - 1,
  _OWNER_END = std::numeric_limits<rank_type>::max() - 1
};

template <typename T> struct DTYPE {};
template <> struct DTYPE<double> { constexpr static DTypeId value = FLOAT64; };
template <> struct DTYPE<float> { constexpr static DTypeId value = FLOAT32; };
template <> struct DTYPE<int64_t> { constexpr static DTypeId value = INT64; };
template <> struct DTYPE<int32_t> { constexpr static DTypeId value = INT32; };
template <> struct DTYPE<int16_t> { constexpr static DTypeId value = INT16; };
template <> struct DTYPE<int8_t> { constexpr static DTypeId value = INT8; };
template <> struct DTYPE<uint64_t> { constexpr static DTypeId value = UINT64; };
template <> struct DTYPE<uint32_t> { constexpr static DTypeId value = UINT32; };
template <> struct DTYPE<uint16_t> { constexpr static DTypeId value = UINT16; };
template <> struct DTYPE<uint8_t> { constexpr static DTypeId value = UINT8; };
template <> struct DTYPE<bool> { constexpr static DTypeId value = BOOL; };

template <DTypeId DT> struct TYPE {};
template <> struct TYPE<FLOAT64> {
  using dtype = double;
  static constexpr bool is_integral = false;
  static constexpr bool is_float = true;
};
template <> struct TYPE<FLOAT32> {
  using dtype = float;
  static constexpr bool is_integral = false;
  static constexpr bool is_float = true;
};
template <> struct TYPE<INT64> {
  using dtype = int64_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<INT32> {
  using dtype = int32_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<INT16> {
  using dtype = int16_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<INT8> {
  using dtype = int8_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<UINT64> {
  using dtype = uint64_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<UINT32> {
  using dtype = uint32_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<UINT16> {
  using dtype = uint16_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<UINT8> {
  using dtype = uint8_t;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};
template <> struct TYPE<BOOL> {
  using dtype = bool;
  static constexpr bool is_integral = true;
  static constexpr bool is_float = false;
};

[[maybe_unused]] static size_t sizeof_dtype(const DTypeId dt) {
  switch (dt) {
  case FLOAT64:
  case INT64:
  case UINT64:
    return 8;
  case FLOAT32:
  case INT32:
  case UINT32:
    return 4;
  case INT16:
  case UINT16:
    return 2;
  case INT8:
  case UINT8:
    return 1;
  case BOOL:
    return sizeof(bool);
  default:
    throw std::runtime_error("unknown dtype");
  };
};

static bool is_float(DTypeId t) {
  switch (t) {
  case SHARPY::DTypeId::FLOAT64:
  case SHARPY::DTypeId::FLOAT32:
    return true;
  default:
    return false;
  }
}

static bool is_int(DTypeId t) {
  switch (t) {
  case SHARPY::DTypeId::INT64:
  case SHARPY::DTypeId::INT32:
  case SHARPY::DTypeId::INT16:
  case SHARPY::DTypeId::INT8:
    return true;
  default:
    return false;
  }
}

static bool is_uint(DTypeId t) {
  switch (t) {
  case SHARPY::DTypeId::UINT64:
  case SHARPY::DTypeId::UINT32:
  case SHARPY::DTypeId::UINT16:
  case SHARPY::DTypeId::UINT8:
  case SHARPY::DTypeId::BOOL:
    return true;
  default:
    return false;
  }
}

static size_t dtype_bitwidth(DTypeId t) {
  switch (t) {
  case SHARPY::DTypeId::FLOAT64:
  case SHARPY::DTypeId::INT64:
  case SHARPY::DTypeId::UINT64:
    return 64;
  case SHARPY::DTypeId::FLOAT32:
  case SHARPY::DTypeId::INT32:
  case SHARPY::DTypeId::UINT32:
    return 32;
  case SHARPY::DTypeId::INT16:
  case SHARPY::DTypeId::UINT16:
    return 16;
  case SHARPY::DTypeId::INT8:
  case SHARPY::DTypeId::UINT8:
    return 8;
  case SHARPY::DTypeId::BOOL:
    return 1;
  default:
    assert(!"Unknown DTypeId");
  }
}

static DTypeId get_float_dtype(size_t bitwidth) {
  switch (bitwidth) {
  case 64:
    return SHARPY::DTypeId::FLOAT64;
  case 32:
    return SHARPY::DTypeId::FLOAT32;
  default:
    assert(!"Unknown bitwidth");
  }
}

static DTypeId get_int_dtype(size_t bitwidth) {
  switch (bitwidth) {
  case 64:
    return SHARPY::DTypeId::INT64;
  case 32:
    return SHARPY::DTypeId::INT32;
  case 16:
    return SHARPY::DTypeId::INT16;
  case 8:
    return SHARPY::DTypeId::INT8;
  default:
    assert(!"Unknown bitwidth");
  }
}

[[maybe_unused]] static DTypeId promoted_dtype(DTypeId a, DTypeId b) {
  if ((is_float(a) && is_float(b)) || (is_int(a) && is_int(b)) ||
      (is_uint(a) && is_uint(b))) {
    return dtype_bitwidth(a) > dtype_bitwidth(b) ? a : b;
  }
  if (is_float(a) || is_float(b)) {
    return get_float_dtype(std::max(dtype_bitwidth(a), dtype_bitwidth(b)));
  }
  // mixed signed/unsigned int case
  size_t si_width, ui_width, max_width = 64;
  if (is_uint(a)) {
    ui_width = dtype_bitwidth(a);
    si_width = dtype_bitwidth(b);
  } else {
    ui_width = dtype_bitwidth(b);
    si_width = dtype_bitwidth(a);
  }
  if (ui_width < si_width) {
    return get_int_dtype(si_width);
  }
  return get_int_dtype(std::min(2 * ui_width, max_width));
}

using RedOpType = ReduceOpId;

inline RedOpType red_op(const char *op) {
  if (!strcmp(op, "max"))
    return MAX;
  if (!strcmp(op, "min"))
    return MIN;
  if (!strcmp(op, "sum"))
    return SUM;
  if (!strcmp(op, "prod"))
    return PROD;
  if (!strcmp(op, "mean"))
    return MEAN;
  if (!strcmp(op, "std"))
    return STD;
  if (!strcmp(op, "var"))
    return VAR;
  throw std::logic_error("unsupported reduction operation");
}

inline shape_type reduce_shape(const shape_type &shape,
                               const dim_vec_type &dims) {
  auto ssz = shape.size();
  auto nd = dims.size();

  if (nd == 0 || nd == ssz)
    return shape_type{};

  shape_type shp(ssz - nd);
  if (shp.size()) {
    int p = -1;
    for (auto i = 0ul; i < ssz; ++i) {
      if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
        shp[++p] = shape[i];
      }
    }
  }
  return shp;
}

template <typename T> typename T::value_type VPROD(const T &v) {
  return std::accumulate(v.begin(), v.end(), 1,
                         std::multiplies<typename T::value_type>());
}

template <typename V>
std::string to_string(const std::vector<V> &vals, char sep = ' ') {
  std::string s = "{";
  if (!vals.empty()) {
    auto _x = vals.begin();
    s += std::to_string(*_x);
    for (++_x; _x != vals.end(); ++_x)
      s += sep + std::to_string(*_x);
  }
  s += "}";
  return s;
}

using id_type = uint64_t;

enum FactoryId : int {
  F_ARANGE,
  F_EWBINOP,
  F_EWUNYOP,
  F_FROMLOCALS,
  F_FULL,
  F_GATHER,
  F_GETITEM,
  F_GETLOCALS,
  F_IEWBINOP,
  F_LINALGOP,
  F_LINSPACE,
  F_MAP,
  F_RANDOM,
  F_REDUCEOP,
  F_REPLICATE,
  F_RESHAPE,
  F_SERVICE,
  F_SETITEM,
  F_ASTYPE,
  F_TODEVICE,
  FACTORY_LAST
};

// size of memreftype in number of intptr_t's
inline uint64_t memref_sz(int rank) { return 3 + 2 * rank; }
inline uint64_t ndarray_sz(int rank, bool isDist) {
  return memref_sz(rank) + isDist ? 2 * memref_sz(rank) + memref_sz(1) : 0;
};
} // namespace SHARPY
