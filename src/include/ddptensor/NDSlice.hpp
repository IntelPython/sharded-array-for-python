// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <bitsery/bitsery.h>
#include <bitsery/traits/vector.h>

///
/// A slice of n-dimensional range with utility features to extract nd-indices.
/// Represented as a vector of triplets [start, end, step[.
///
class NDSlice {
public:
  using vec_t = std::vector<uint64_t>;

private:
  // vector with offsets per dimension
  vec_t _offsets;
  // vector with sizes per dimension
  vec_t _sizes;
  // vector with strides per dimension
  vec_t _strides;

public:
  NDSlice() = default;
  NDSlice(NDSlice &&) = default;
  NDSlice(const NDSlice &) = default;
  // assumes a function exists to extract values per slice (compute_slice(T,
  // ...))
  template <typename T>
  NDSlice(const std::vector<T> &v)
      : _offsets(v.size()), _sizes(v.size()), _strides(v.size()) {
    auto nd = v.size();
    auto i = 0;
    for (auto s : v) {
      compute_slice(s, _offsets[i], _sizes[i], _strides[i]);
      ++i;
    };
  }

  const vec_t &offsets() const { return _offsets; };
  const vec_t &sizes() const { return _sizes; };
  const vec_t &strides() const { return _strides; };
  const uint64_t size() const { return VPROD(_sizes); };

  template <typename S> void serialize(S &ser) {
    ser.container8b(_offsets, 8);
    ser.container8b(_sizes, 8);
    ser.container8b(_strides, 8);
  }
};

#if 0
    NDSlice(const Slice & slc) : _slice_vec(1, slc), _sizes() {}
    NDSlice(const NDSlice & o) : _slice_vec(o._slice_vec), _sizes() {}
    NDSlice(const std::initializer_list<Slice> & slc) : _slice_vec(slc), _sizes() {}
    NDSlice(const ndslice_t & sv) : _slice_vec(sv), _sizes() {}
    NDSlice(ndslice_t && sv) : _slice_vec(std::move(sv)), _sizes() {}
    template<typename T>
    NDSlice(const std::vector<T> & v)
        : _slice_vec(v.size()), _sizes()
    {
        uint32_t i = -1;
        for(auto s : v) _slice_vec[++i] = Slice(s);
    }
    NDSlice(const NDIndex & idx)
        : _slice_vec(idx.size()), _sizes()
    {
        uint32_t i = -1;
        for(auto s : idx) _slice_vec[++i] = Slice(s, s+1);
    }

    NDSlice & operator=(NDSlice &&) = default;
    NDSlice & operator=(const NDSlice &) = default;

    // inits helper attribute _sizes (for lazy computation)
    // [i] holds the number of elements for dimensions [i,N[
    void init_sizes() const
    {
        _sizes.resize(_slice_vec.size());
        auto v = _sizes.rbegin();
        value_type::value_type sz = 1;
        for(auto i = _slice_vec.rbegin(); i != _slice_vec.rend(); ++i, ++v) {
            sz *= (*i).size();
            *v = sz;
        }
    }

    ///
    /// @return number of elements for each dimension-slice as a vector
    ///
    vec_t shape() const
    {
        std::vector<uint64_t> ret(_slice_vec.size());
        auto v = ret.begin();
        for(auto i = _slice_vec.begin(); i != _slice_vec.end(); ++i, ++v) {
            *v = (*i).size();
        }
        return ret;
    }

    std::vector<Slice> slices() const
    {
        return _slice_vec;
    }

    ///
    /// @return total number of elements represented by the nd-slice
    ///
    value_type::value_type size(uint64_t dim = 0) const
    {
        if(_slice_vec.empty()) return 0;
        if(_sizes.empty()) init_sizes();
        return _sizes[dim];
    }

    ///
    /// @return number of dimensions
    ///
    size_t ndims() const
    {
        return _slice_vec.size();
    }

    ///
    /// Set slice of given dimension to slc
    ///
    void set(uint64_t dim, const Slice & slc)
    {
        _slice_vec[dim] = slc;
        _sizes.resize(0);
    }

    template<typename C>
    NDSlice _convert(const C & conv) const
    {
        ndslice_t sv;
        sv.reserve(_slice_vec.size());
        for(auto i = 0; i != _slice_vec.size(); ++i) {
            sv.push_back(conv(i));
        }
        return NDSlice(std::move(sv));
    }

    ///
    /// @return Subslice mapped into this' index space
    ///
    NDSlice slice(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].slice(slc.dim(i));
            } );
    }

    NDSlice normalize() const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].normalize();
            } );
    }

    ///
    /// @return NDSlice with same same, shifted left by _start in each dimension and with strides 1
    ///
    NDSlice deflate(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].deflate(slc.dim(i)._start);
            } );
    }

    ///
    /// @return Slice with same same, shifted right by off and with stride strd
    ///
    NDSlice inflate(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].inflate(slc.dim(i)._start, slc.dim(i)._step);
            } );
    }

    ///
    /// @return Copy of NDSlice which was shifted left by _start in each dim of slc
    ///
    NDSlice shift(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].shift(slc.dim(i)._start);
            } );
    }

    ///
    /// @return Copy of NDSlice which was trimmed by given slice
    ///
    NDSlice trim(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].trim(slc.dim(i)._start, slc.dim(i)._end);
            } );
    }

    ///
    /// @return Copy of NDSlice which was trimmed by given slice
    ///
    NDSlice overlap(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].overlap(slc.dim(i));
            } );
    }

    ///
    /// @return Copy of NDSlice which was trimmed by t_slc and shifted by s_slc._start's
    ///
    NDSlice trim_shift(const NDSlice & t_slc, const NDSlice & s_slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].trim(t_slc.dim(i)._start, t_slc.dim(i)._end).shift(s_slc.dim(i)._start);
            } );
    }

    NDSlice map(const NDSlice & slc) const
    {
        return _convert([&](uint64_t i) {
                return _slice_vec[i].map(slc.dim(i));
            } );
    }

    NDSlice normalize(uint32_t D) const
    {
        ndslice_t slc(_slice_vec);
        slc[D] = slc[D].normalize();
        return NDSlice(slc);
    }

    ///
    /// @return NDSlice with same same, shifted left by offset in dimension D and with strides 1
    ///
    NDSlice deflate(uint32_t D, uint64_t offset) const
    {
        ndslice_t slc(_slice_vec);
        slc[D] = slc[D].deflate(offset);
        return NDSlice(slc);
    }

    ///
    /// @return Copy of NDSlice which was trimmed in D'th dimension
    ///
    NDSlice trim(uint32_t D, const value_type::value_type start, const value_type::value_type end) const
    {
        ndslice_t slc(_slice_vec);
        slc[D] = slc[D].trim(start, end);
        return NDSlice(std::move(slc));
    }

    ///
    /// @return Copy of NDSlice which was shifted left in D'th dimension
    ///
    NDSlice shift(uint32_t D, value_type::value_type s) const
    {
        ndslice_t slc(_slice_vec);
        slc[D] = slc[D].shift(s);
        return NDSlice(std::move(slc));
    }

    ///
    /// @return Copy of NDSlice which was trimmed and shifted left in D'th dimension
    ///
    NDSlice trim_shift(uint32_t D, const value_type::value_type start, const value_type::value_type end, const value_type::value_type shft) const
    {
        ndslice_t slc(_slice_vec);
        slc[D] = slc[D].trim(start, end).shift(shft);
        return NDSlice(std::move(slc));
    }

    ///
    /// @return Number of elements if slice ended at given pos in 1st dimension
    ///
    value_type::value_type size_upto(value_type::value_type end) const
    {
        ndslice_t slc(_slice_vec);
        slc[0] = Slice(slc[0]._start, end, slc[0]._step);
        return NDSlice(std::move(slc)).size();
    }

    ///
    /// @return Slice for dim d
    ///
    const ndslice_t::value_type & dim(size_t d) const
    {
        return _slice_vec[d];
    }

    ///
    /// @return true if idx is in ndslice
    ///
    bool covers(const value_type & idx) const
    {
        auto i = idx.begin();
        for(auto x : _slice_vec) {
            if(!x.covers(*i)) return false;
            ++i;
        }
        return true;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.container(_slice_vec, 8);
    }

    friend std::ostream &operator<<(std::ostream &output, const NDSlice & slc) {
        output << "[";
        for(auto x : slc._slice_vec) output << x << ", ";
        output << "]";
        return output;
    }

    /// STL-like iterator for iterating through the pvslice (flattened index space).
    /// Supports pre-increment ++, unequality check != and dereference *
    class iterator {
        // we keep a reference to the ndslice
        const NDSlice * _ndslice;
        // and the current position
        NDIndex _curr_pos;
        // and the current flattened index
        uint64_t _curr_idx;

    public:
        iterator(const NDSlice * ndslice = nullptr)
            : _ndslice(ndslice), _curr_pos(ndslice ? ndslice->ndims() : 0), _curr_idx(-1)
        {
            if(ndslice) {
                _curr_idx = 0;
                auto const bshp = _ndslice->shape();
                uint64_t tsz = 1;
                for(int64_t d = _ndslice->ndims()-1; d >= 0; --d) {
                    auto const & cs = _ndslice->dim(d);
                    _curr_pos[d] = cs._start;
                    _curr_idx += cs._start * tsz;
                    tsz *= bshp[d];
                }
            }
        }

        iterator& operator++() noexcept
        {
            auto const bshp = _ndslice->shape();
            uint64_t tsz = 1;
            for(int64_t d = _ndslice->ndims()-1; d >= 0; --d) {
                auto const & cs = _ndslice->dim(d);
                auto x = _curr_pos[d] + cs._step;
                if(x < cs._end) {
                    _curr_pos[d] = x;
                    _curr_idx += cs._step * tsz;
                    return *this;
                }
                // rewind: next round we'll increment the full tsz
                _curr_idx -= (_curr_pos[d] - cs._start) * tsz;
                _curr_pos[d] = cs._start;
                tsz *= bshp[d];
            }
            *this = iterator();
            return *this;
        }

        // FIXME: opt for performance
        iterator& operator+=(uint64_t inc) noexcept
        {
            auto _end = iterator();
            while(inc > 0) {
                ++(*this);
                --inc;
                if(*this == _end) return *this;
            }
            return *this;
        }

        bool operator!=(const iterator & other) const noexcept
        {
            return  _curr_idx != other._curr_idx || _ndslice != other._ndslice;
        }

        bool operator==(const iterator & other) const noexcept
        {
            return !operator!=(other);
        }

        uint64_t operator*() const noexcept
        {
            return _curr_idx;
        }

        // capacity is number of elements
        template<typename F, typename T>
        NDSlice::iterator& fill_buffer(T * buff, uint64_t capacity, uint64_t & ncopied, const F * from) noexcept
        {
            ncopied = 0;
            uint64_t n = capacity;
            uint64_t i=0;
            for(; ncopied<n && (*this) != _ndslice->end(); ++ncopied) {
                buff[ncopied] = static_cast<T>(from[*(*this)]);
                ++(*this);
            }
            return *this;
        }
    };

    ///
    /// @return STL-like Iterator pointing to first element
    ///
    iterator begin() const noexcept
    {
        return size() ? iterator(this) : end();
    }

    ///
    /// @return STL-like Iterator pointing to first element after end
    ///
    iterator end() const noexcept
    {
        return iterator();
    }
};
#endif // if 0

#if 0
    // ###########################################################################
    // ###########################################################################
    ///
    /// A STL-like iterator for iterating through a nd-slice, returning NDIndex
    /// Supports pre-increment ++, unequality check != and dereference *
    ///
    struct iterator {
        typedef std::vector<std::pair<Slice, Slice::iterator>> NDIterator;

        iterator(const ndslice_t * ndslice = nullptr)
            : ndslice_(ndslice), curr_(0), curr_idx_(0) {
            if(ndslice) {
                auto slc = ndslice_->begin();
                assert(slc != ndslice_->end());
                curr_idx_.resize(ndslice->size(), 0);
                auto i = curr_idx_.begin();
                for(; slc != ndslice_->end(); ++slc, ++i) {
                    curr_.push_back({*slc, (*slc).begin()});
                    *i = *(*slc).begin();
                }
            }
        }

        iterator& operator++() noexcept
        {
            auto v = curr_idx_.rbegin();
            auto x = 0;
            for(auto d = curr_.rbegin(); d != curr_.rend(); ++d, ++v, ++x) {
                ++(*d).second;
                if((*d).second != (*d).first.end()) {
                    *v = *(*d).second;
                    return *this;
                }
                (*d).second = (*d).first.begin();
                *v = *(*d).second;
            }
            *this = iterator();
            return *this;
        }

        bool operator!=(const iterator & other) const noexcept
        {
            return ndslice_ != other.ndslice_ && curr_idx_ != other.curr_idx_;
        }

        const value_type & operator*() const noexcept
        {
            return curr_idx_;
        }

    private:
        // we keep a reference to the orig nd-slice/vector
        const ndslice_t * ndslice_;
        // and the current position/iterators
        NDIterator curr_;
        // and the current nd-index
        value_type curr_idx_;
    };

    ///
    /// @return STL-like Iterator pointing to first element
    ///
    iterator begin() const noexcept
    {
        return size() ? iterator(&_slice_vec) : end();
    }

    ///
    /// @return STL-like Iterator pointing to first element after end
    ///
    iterator end() const noexcept
    {
        return iterator();
    }
#endif
