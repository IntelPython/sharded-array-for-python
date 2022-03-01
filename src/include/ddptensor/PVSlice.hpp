// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "NDSlice.hpp"
#include "Transceiver.hpp"  // rank_type

using offsets_type = std::vector<uint64_t>;

constexpr static int NOSPLIT = -1;

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
    BasePVSlice(const shape_type & shape, int split=0)
        : _offset(split == NOSPLIT ? 0 : (shape[split] + theTransceiver->nranks() - 1) / theTransceiver->nranks()),
          _shape(shape),
          _split_dim(split)
    {
        _tile_size = VPROD(_shape) / shape[_split_dim] * _offset;
    }

    BasePVSlice(shape_type && shape, int split=0)
        : _offset(split == NOSPLIT ? 0 : (shape[split] + theTransceiver->nranks() - 1) / theTransceiver->nranks()),
          _shape(std::move(shape)),
          _split_dim(split)
    {
        _tile_size = VPROD(_shape) / _shape[_split_dim] * _offset;
    }

    bool is_equally_tiled() const
    {
        return _shape[_split_dim] == theTransceiver->nranks() * _offset;
    }

    uint64_t offset() const
    {
        return _offset;
    }

    uint64_t tile_size(rank_type rank = theTransceiver->rank()) const
    {
        // only rank 0 is guaranteed to have _tile_size, all other parts can be < _tile_size
        if(rank == 0) return _tile_size;
        auto sz = VPROD(_shape);
        auto off = rank * _tile_size;
        if(sz >= off) return _tile_size;
        auto r = off - sz;
        // if r < _tile_size it's the remainder, otherwise we are past the end of the global array
        return r < _tile_size ? r : 0UL;
    }

    shape_type tile_shape(rank_type rank = theTransceiver->rank()) const
    {
        // only rank 0 is guaranteed to have _tile_size, all other parts can be < _tile_size
        shape_type r(_shape);
        if(rank == 0) r[_split_dim] = offset();
        else {
            auto end = (rank+1) * offset();
            if(r[_split_dim] >= end) r[_split_dim] = offset();
            else {
                auto diff = end - r[_split_dim];
                // if diff < offset() it's the remainder, otherwise we are past the end of the global array
                r[_split_dim] = diff < offset() ? diff : 0UL;
            }
        }
        return r;
    }

    int split_dim() const
    {
        return _split_dim;
    }

    const shape_type & shape() const
    {
        return _shape;
    }

    shape_type shape(rank_type rank) const
    {
        if(split_dim() == NOSPLIT) {
            return rank == theTransceiver->rank() ? _shape : shape_type();
        }
        shape_type shp(_shape);
        auto end = (rank+1) * _offset;
        if(end <= _shape[_split_dim]) shp[_split_dim] = _offset;
        else shp[_split_dim] = end - _shape[_split_dim];
        return shp;
    }

    rank_type owner(const NDSlice & slice) const
    {
        return split_dim() == NOSPLIT ? theTransceiver->rank() : slice.dim(split_dim())._start / offset();
    }
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
    PVSlice(const shape_type & shp, int split=0)
        : _slice(shp),
          _base(std::make_shared<BasePVSlice>(shp, split)),
          _shape()
    {
    }

    PVSlice(shape_type && shp, int split=0)
        : _slice(shp), // std::move in next line
          _base(std::make_shared<BasePVSlice>(std::move(shp), split)),
          _shape()
    {
    }

    PVSlice(const shape_type & shp, const NDSlice & slc, int split=0)
        : _slice(slc),
          _base(std::make_shared<BasePVSlice>(shp, split)),
          _shape()
    {
    }

    PVSlice(shape_type && shp, NDSlice && slc, int split=0)
        : _slice(std::move(slc)),
          _base(std::make_shared<BasePVSlice>(std::move(shp), split)),
          _shape()
    {
    }

    PVSlice(const PVSlice & org, const NDSlice & slice)
        : _slice(std::move(org._slice.slice(slice))),
          _base(org._base),
          _shape()
    {
    }

    PVSlice(const PVSlice & org, rank_type rank)
        : _slice(std::move(org.slice_of_rank(rank))),
          _base(org._base),
          _shape()
    {
    }

private:
    PVSlice(BasePVSlicePtr bp, const NDSlice & slice)
        : _slice(slice),
          _base(bp),
          _shape()
    {
    }

public:
    uint64_t ndims() const
    {
        return _slice.ndims();
    }

    const int split_dim() const
    {
        return _base->split_dim();
    }

    const bool is_sliced() const
    {
        return base_shape() != shape();
    }

    bool is_equally_tiled() const
    {
        return _base->is_equally_tiled();
    }

    const uint64_t tile_size(rank_type rank = theTransceiver->rank()) const
    {
        return _base->tile_size(rank);
    }

    const shape_type & shape() const
    {
        if(_shape.size() != _slice.ndims()) {
            _shape.resize(_slice.ndims());
            int i = -1;
            for(auto & shp : _shape) shp = _slice.dim(++i).size();
        }
        return _shape;
    }

    const shape_type tile_shape(rank_type rank = theTransceiver->rank()) const
    {
        return _base->tile_shape(rank);
    }

    const shape_type shape_of_rank(rank_type rank = theTransceiver->rank()) const
    {
        return slice_of_rank(rank).shape();
    }

    const shape_type & base_shape() const
    {
        return _base->shape();
    }

    rank_type owner(const NDSlice & slice) const
    {
        return _base->owner(slice);
    }

    const NDSlice & slice() const
    {
        return _slice;
    }

    uint64_t size() const
    {
        return slice().size();
    }

#if 0
    NDSlice normalized_slice() const
    {
        return _slice.normalize(_base->split_dim());
    }
#endif

    NDSlice map_slice(const NDSlice & slc) const
    {
        return _slice.map(slc);
    }

    NDSlice slice_of_rank(rank_type rank = theTransceiver->rank()) const
    {
        if(_base->split_dim() == NOSPLIT) {
            return rank == theTransceiver->rank() ? slice() : NDSlice();
        }
        return _slice.trim(_base->split_dim(), rank * _base->offset(), (rank+1) * _base->offset());
    }

    NDSlice local_slice_of_rank(rank_type rank = theTransceiver->rank()) const
    {
        if(_base->split_dim() == NOSPLIT) {
            return rank == theTransceiver->rank() ? slice() : NDSlice();
        }
        return _slice.trim_shift(_base->split_dim(),
                                 rank * _base->offset(),
                                 (rank+1) * _base->offset(),
                                 rank * _base->offset());
    }

    bool need_reduce(const dim_vec_type & dims) const
    {
        if(_base->split_dim() == NOSPLIT) return false;
        auto nd = dims.size();
        // Reducing to a single scalar or over a subset of dimensions *including* the split axis.
        if(nd == 0
           || nd == _slice.ndims()
           || std::find(dims.begin(), dims.end(), _base->split_dim()) != dims.end()) return true;

        // *not* reducing over split axis
        return false;
    }  ///

    
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
        iterator(const PVSlice * pvslice = nullptr)
            : _pvslice(pvslice), _curr_pos(pvslice ? pvslice->ndims() : 0), _curr_idx(-1)
        {
            if(pvslice) {
                _curr_idx = 0;
                auto const bshp = _pvslice->base_shape();
                uint64_t tsz = 1;
                for(int64_t d = _pvslice->ndims()-1; d >= 0; --d) {
                    auto const & cs = _pvslice->slice().dim(d);
                    _curr_pos[d] = cs._start;
                    _curr_idx += cs._start * tsz;
                    tsz *= bshp[d];
                }
            }
        }

        iterator& operator++() noexcept
        {
            auto const bshp = _pvslice->base_shape();
            uint64_t tsz = 1;
            for(int64_t d = _pvslice->ndims()-1; d >= 0; --d) {
                auto const & cs = _pvslice->slice().dim(d);
                auto x = _curr_pos[d] + cs._step;
                if(x < cs._end) {
                    _curr_pos[d] = x;
                    _curr_idx += cs._step * tsz;
                    return *this;
                }
                _curr_idx += (bshp[d] - (_curr_pos[d] - cs._start)) * tsz;
                _curr_pos[d] = cs._start;
                tsz *= bshp[d];
            }
            *this = iterator();
            return *this;
        }

        bool operator!=(const iterator & other) const noexcept
        {
            return  _pvslice != other._pvslice && _curr_idx != other._curr_idx;
        }

        uint64_t operator*() const noexcept
        {
            return _curr_idx;
        }
    };

    ///
    /// @return STL-like Iterator pointing to first element
    ///
    iterator begin() const noexcept
    {
        return slice().size() ? iterator(this) : end();
    }

    ///
    /// @return STL-like Iterator pointing to first element after end
    ///
    iterator end() const noexcept
    {
        return iterator();
    }

    friend std::ostream &operator<<(std::ostream &output, const PVSlice & slc) {
        output << "{slice=" << slc.slice()
               << "base=" << to_string(slc._base->shape())
               << "offset=" << slc._base->offset()<< "}";
        return output;
    }
};
