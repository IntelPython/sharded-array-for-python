// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

///
/// Multi-dimensional index
///
typedef std::vector<int64_t> NDIndex;

#if 0
///
/// @return tile-sizes for each dimension, as if leading dimensions were cut.
/// @param tile_shape tile-shape in question
///
template<typename T>
static std::vector<uint64_t> tile_sub_sizes(const std::vector<T> & tile_shape)
{
    std::vector<uint64_t> szs(tile_shape.size(), 0);
    szs.back() = tile_shape.back();
    for(uint64_t i = 2; i <= tile_shape.size(); ++i) {
        auto idx = tile_shape.size() - i;
        szs[idx] = tile_shape[idx] * szs[idx+1];
    }
    return szs;
}

///
/// @return pair of (rank, local linearized (1d) index for global nd-index)
/// @param idx The index
/// @param tile_shape tile-shape to assume
/// @param tssizes helper for more efficient computatino, see tile_sub_sizes
///
template<typename T>
uint64_t linearize(const std::vector<T> & idx, const std::vector<uint64_t> & tssizes)
{
    uint64_t tidx = idx.back();
    auto i = idx.rbegin()+1;
    auto sz = tssizes.rbegin();
    for(; i != idx.rend(); ++sz, ++i) {
        tidx += (*sz) * (*i);
    }
    return tidx;
}
#endif
