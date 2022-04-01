// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <array>
#include "NDSlice.hpp"
#include "Transceiver.hpp"  // rank_type

using offsets_type = std::vector<uint64_t>;

constexpr static int NOSPLIT = 1;

class BasePVSlice
{
    uint64_t   _offset;
    uint64_t   _tile_size;
    shape_type _shape;
    int        _split_dim;

public:
    BasePVSlice() = delete;
    BasePVSlice(const BasePVSlice &) = delete;
    BasePVSlice(BasePVSlice &&) = default;
    BasePVSlice(const shape_type & shape, int split=0);
    BasePVSlice(shape_type && shape, int split=0);
    bool is_equally_tiled() const;
    uint64_t offset() const;
    uint64_t tile_size(rank_type rank = theTransceiver->rank()) const;
    shape_type tile_shape(rank_type rank = theTransceiver->rank()) const;
    int split_dim() const;
    const shape_type & shape() const;
    rank_type owner(const NDSlice & slice) const;
};

using BasePVSlicePtr = std::shared_ptr<BasePVSlice>;

class PVSlice
{
    NDSlice     _slice;  // must go before _base
    BasePVSlicePtr _base;   
    mutable shape_type  _shape;

public:
    PVSlice() = delete;
    PVSlice(const PVSlice &) = default;
    PVSlice(PVSlice &&) = default;
    PVSlice(const shape_type & shp, int split=0);
    PVSlice(shape_type && shp, int split=0);
    PVSlice(const shape_type & shp, const NDSlice & slc, int split=0);
    PVSlice(shape_type && shp, NDSlice && slc, int split=0);
    PVSlice(const PVSlice & org, const NDSlice & slice);
    PVSlice(const PVSlice & org, rank_type rank);
private:
    PVSlice(BasePVSlicePtr bp, const NDSlice & slice);

public:
    uint64_t ndims() const;
    int split_dim() const;
    bool is_sliced() const;
    bool local_is_contiguous(rank_type rank = theTransceiver->rank()) const;
    bool is_equally_tiled() const;
    rank_type owner(const NDSlice & slice) const;
    const shape_type & base_shape() const;
    // global slice
    const NDSlice & slice() const;
    // global shape
    const shape_type & shape() const;
    // global size
    uint64_t size() const;
    // size of rank's tile, e.g. the allocated memory
    uint64_t tile_size(rank_type rank = theTransceiver->rank()) const;
    // shape of rank's tile, e.g. the allocated memory
    shape_type tile_shape(rank_type rank = theTransceiver->rank()) const;
    // ranks's slice of local tile
    NDSlice tile_slice(rank_type rank = theTransceiver->rank()) const;
    // size of rank's slice
    // size of rank's slice
    uint64_t local_size(rank_type rank = theTransceiver->rank()) const;
    // shape of rank's slice
    shape_type local_shape(rank_type rank = theTransceiver->rank()) const;
    // shape of rank's slice
    NDSlice local_slice(rank_type rank = theTransceiver->rank()) const;
    NDSlice map_slice(const NDSlice & slc) const;
    // Compute overlapping slices for every (rank,rank) pair for mapping o_slc onto this.
    // Returns a 2 lists of size number-of-ranks:
    //   0. which local slice to send to rank i
    //   1. which local slice maps to a slice of o_slc on rank i
    std::array<std::vector<NDSlice>, 2> map_ranks(const PVSlice & o_slc) const;
    bool need_reduce(const dim_vec_type & dims) const;

    friend std::ostream &operator<<(std::ostream &output, const PVSlice & slc);
};
