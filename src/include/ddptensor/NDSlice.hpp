// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <bitsery/bitsery.h>
#include <bitsery/traits/vector.h>

#include "Slice.hpp"
#include "NDIndex.hpp"

///
/// A slice of n-dimensional range with utility features to extract nd-indices.
/// Represented as a vector of triplets [start, end, step[.
///
class NDSlice {
    // type of elements, n-dimensional
    typedef std::vector<Slice::value_type> value_type;
    // the nd-slice, e.g. once slice per dimension
    typedef std::vector<Slice> ndslice_t;


    // vector with one Slice per dimension, e.g. our nd-slice
    ndslice_t _slice_vec;
    // helper, holding sizes of each dimensions's slice
    mutable std::vector<uint64_t> _sizes;

public:
    NDSlice() = default;
    NDSlice(NDSlice &&) = default;
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
    /// @return true if NDSlice is a lbock, e.g. all steps == 1
    ///
    bool is_block() const
    {
        for(auto s : _slice_vec) {
            if(! s.is_block()) return false;
        }
        return true;
    }

    ///
    /// @return number of elements for each dimension-slice as a vector
    ///
    shape_type shape() const
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
    value_type::value_type size() const
    {
        if(_sizes.empty()) init_sizes();
        return _sizes[0];
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

    ///
    /// @return ith index-tuple in canonical (flat) order of the expanded slice.
    /// does not check bounds, e.g. can return indices beyond end of slice
    ///
    value_type operator[](value_type::value_type i) const {
        if(_sizes.empty()) init_sizes();
        value_type ret(_slice_vec.size(), 0);
        auto sz = ++(_sizes.begin());
        auto slc = _slice_vec.rbegin();
        // iterate over dimensions to compute ith index
        for(auto v = ret.begin(); v != ret.end(); ++v, ++slc) {
            if(sz != _sizes.end()) {
                auto idx = i / (*sz);
                *v = (*slc)[idx];
                i -= idx * (*sz);
                ++sz;
            } else {
                *v = (*slc)[i];
            }
        }
        return ret;
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
};

inline py::tuple _make_tuple(const NDSlice & v)
{
    using V = NDSlice;
    return _make_tuple(v, [](const V & v){return v.ndims();}, [](const V & v, int i){return v.dim(i).pyslice();});
}
