// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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
    const int split_dim() const;
    const bool is_sliced() const;
    bool is_equally_tiled() const;
    const uint64_t tile_size(rank_type rank = theTransceiver->rank()) const;
    const shape_type & shape() const;
    const shape_type tile_shape(rank_type rank = theTransceiver->rank()) const;
    const shape_type & base_shape() const;
    rank_type owner(const NDSlice & slice) const;
    const NDSlice & slice() const;
    uint64_t size() const;
    NDSlice map_slice(const NDSlice & slc) const;
    NDSlice slice_of_rank(rank_type rank = theTransceiver->rank()) const;
    NDSlice local_slice_of_rank(rank_type rank = theTransceiver->rank()) const;
    bool need_reduce(const dim_vec_type & dims) const;

    /// STL-like iterator for iterating through the pvslice.
    /// Supports pre-increment ++, unequality check != and dereference *
    ///
    class iterator {
        // we keep a reference to the pvslice
        const PVSlice * _pvslice;
        // and the current position
        NDIndex _curr_pos;
        // and the current flattened index
        uint64_t _curr_idx;

    public:
        iterator(const PVSlice * pvslice = nullptr);
        iterator& operator++() noexcept;
        bool operator!=(const iterator & other) const noexcept;
        uint64_t operator*() const noexcept;
    };

    ///
    /// @return STL-like Iterator pointing to first element
    ///
    iterator begin() const noexcept;

    ///
    /// @return STL-like Iterator pointing to first element after end
    ///
    iterator end() const noexcept;

    friend std::ostream &operator<<(std::ostream &output, const PVSlice & slc);
};
