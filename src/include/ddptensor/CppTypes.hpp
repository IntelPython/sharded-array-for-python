// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "p2c_ids.hpp"

#include <vector>
#include <numeric>
#include <cstring>

#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>


using shape_type = std::vector<uint64_t>;
using dim_vec_type = std::vector<int>;
using rank_type = uint64_t;
using Buffer = std::vector<uint8_t>;
using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;
using InputAdapter = bitsery::InputBufferAdapter<Buffer>;
using Serializer = bitsery::Serializer<OutputAdapter>;
using Deserializer = bitsery::Deserializer<InputAdapter>;

enum _RANKS: rank_type {
    NOOWNER    = std::numeric_limits<rank_type>::max(),
    REPLICATED = std::numeric_limits<rank_type>::max() - 1,
    _OWNER_END = std::numeric_limits<rank_type>::max() - 1
};

template<typename T> struct DTYPE {};
template<> struct DTYPE<double>   { constexpr static DTypeId value = FLOAT64; };
template<> struct DTYPE<float>    { constexpr static DTypeId value = FLOAT32; };
template<> struct DTYPE<int64_t>  { constexpr static DTypeId value = INT64; };
template<> struct DTYPE<int32_t>  { constexpr static DTypeId value = INT32; };
template<> struct DTYPE<int16_t>  { constexpr static DTypeId value = INT16; };
template<> struct DTYPE<int8_t>   { constexpr static DTypeId value = INT8; };
template<> struct DTYPE<uint64_t> { constexpr static DTypeId value = UINT64; };
template<> struct DTYPE<uint32_t> { constexpr static DTypeId value = UINT32; };
template<> struct DTYPE<uint16_t> { constexpr static DTypeId value = UINT16; };
template<> struct DTYPE<uint8_t>  { constexpr static DTypeId value = UINT8; };
template<> struct DTYPE<bool>     { constexpr static DTypeId value = BOOL; };

template<DTypeId DT> struct TYPE {};
template<> struct TYPE<FLOAT64> { using dtype = double; };
template<> struct TYPE<FLOAT32> { using dtype = float; };
template<> struct TYPE<INT64>   { using dtype = int64_t; };
template<> struct TYPE<INT32>   { using dtype = int32_t; };
template<> struct TYPE<INT16>   { using dtype = int16_t; };
template<> struct TYPE<INT8>    { using dtype = int8_t; };
template<> struct TYPE<UINT64>  { using dtype = uint64_t; };
template<> struct TYPE<UINT32>  { using dtype = uint32_t; };
template<> struct TYPE<UINT16>  { using dtype = uint16_t; };
template<> struct TYPE<UINT8>   { using dtype = uint8_t; };
template<> struct TYPE<BOOL>    { using dtype = bool; };

static size_t sizeof_dtype(const DTypeId dt) {
    switch(dt) {
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

using RedOpType = ReduceOpId;

inline RedOpType red_op(const char * op)
{
    if(!strcmp(op, "max")) return MAX;
    if(!strcmp(op, "min")) return MIN;
    if(!strcmp(op, "sum")) return SUM;
    if(!strcmp(op, "prod")) return PROD;
    if(!strcmp(op, "mean")) return MEAN;
    if(!strcmp(op, "std")) return STD;
    if(!strcmp(op, "var")) return VAR;
    throw std::logic_error("unsupported reduction operation");
}

inline shape_type reduce_shape(const shape_type & shape, const dim_vec_type & dims)
{
    auto ssz = shape.size();
    auto nd = dims.size();

    if(nd == 0 || nd == ssz) return shape_type{};

    shape_type shp(ssz - nd);
    if(shp.size()) {
        int p = -1;
        for(auto i = 0; i < ssz; ++i) {
            if(std::find(dims.begin(), dims.end(), i) == dims.end()) {
                shp[++p] = shape[i];
            }
        }
    }
    return shp;
}

template<typename T>
typename T::value_type VPROD(const T & v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<typename T::value_type>());
}

template <typename V>
std::string to_string(const std::vector<V>& vals, char sep=' ') {
    std::string s = "{";
    if(!vals.empty()) {
        auto _x = vals.begin();
        s += std::to_string(*_x);
        for(++_x; _x != vals.end(); ++_x) s += sep + std::to_string(*_x);
    }
    s += "}";
    return s;
}

using id_type = uint64_t;

enum FactoryId : int {
    F_ARANGE,
    F_FROMSHAPE,
    F_FULL,
    F_UNYOP,
    F_EWUNYOP,
    F_IEWBINOP,
    F_EWBINOP,
    F_REDUCEOP,
    F_MANIPOP,
    F_LINALGOP,
    F_GETITEM,
    F_SETITEM,
    F_RANDOM,
    F_SERVICE,
    F_TONUMPY,
    FACTORY_LAST
};