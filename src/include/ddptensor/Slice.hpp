// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <bitsery/bitsery.h>

///
/// A slice of 1-dimensional range with utility features to extract indices.
/// Represented as a triplet [start, end, step[.
///
struct Slice
{
    // the type of elements, e.g. indices
    typedef ssize_t value_type;
    // the slice representation
    value_type _start, _end, _step;

    Slice()
        : _start(0), _end(-1), _step(1)
    {}

    ///
    /// Construct a slice/range [0:end:1[
    /// @param end first index out of range
    ///
    Slice(value_type end)
        : _start(0), _end(end), _step(1)
    {
    }

    ///
    /// Construct a slice [start:end:step[
    /// @param start first index of range
    /// @param end first index out of range
    /// @param step step between values in the slice
    ///
    Slice(value_type start, value_type end, value_type step=1)
        : _start(start), _end(end), _step(step)
    {
    }

    ///
    /// Construct a slice from a py::slice
    ///
    Slice(const py::slice & slc, int64_t len=std::numeric_limits<int64_t>::max())
    {
        ssize_t slc_len;
        slc.compute(len, &_start, &_end, &_step, &slc_len);
    }

    ///
    /// @return Subslice mapped into this' index space
    ///
    Slice slice(const Slice & slice) const
    {
        return {_start + slice._start * _step, _start + slice._end * _step, _step * slice._step};
    }

    ///
    /// @return py::slice representation
    ///
    py::slice pyslice() const
    {
        return py::slice(_start, _end, _step);
    }

    Slice normalize() const
    {
        return {0, size(), 1};
    }

    ///
    /// @return Slice with same shape, shifted left by off ignorig index space
    ///         and then normalized to stride 1
    /// Note: (_end-off) is the new zero before scaling down. _start will also be scaled down by stride_
    ///
    Slice deflate(value_type off = 0) const
    {
        auto strt = _start - off;
        return {strt/_step, strt + size(), 1};
        // if(_step > 1) return {_start/_step, (_end+_step-1)/_step, 1};
        // return {_start, _end, _step};
    }

    ///
    /// @return Slice with same shape, shifted right by off and with stride strd
    ///
    Slice inflate(value_type off, value_type strd) const
    {
        auto strt = _start + off;
        return {strt, strt + size() * strd, strd};
        // if(_step > 1) return {_start*strd, (_end+_step-1)/_step, 1};
        // return {_start, _end, _step};
    }

    ///
    /// @return copy of this Slice, trimmed to given start and end.
    /// @param s Start index to trim to
    /// @param e End index to trim to
    ///
    Slice trim(value_type s, value_type e) const
    {
        assert(_step > 0);
        auto start = _start;
        if(s > _start) {
            auto m = (s - start) % _step;
            if(m) start = s + _step - m;
            else start = s;
        }
        auto end = std::min(e, _end);
        return {std::min(start, end), end, _step};
    }

    Slice map(const Slice & slc) const
    {
        value_type diff = slc._start - _start;
        if(diff < 0) {
            assert(slc.size() == 0);
            return {0, 0, 1};
        }
        value_type start = diff / _step;
        assert(diff % _step == 0);
        assert((slc._step == _step));
        return {start, start + slc.size(), 1};
    }

    Slice overlap(const Slice & slc) const
    {
        assert(_step == slc._step);
        return trim(slc._start, slc._end);
    }

    ///
    /// @return copy of this Slice, shifted left by given amount.
    ///
    Slice shift(value_type s) const
    {
        return {_start - s, _end - s, _step};
    }

    ///
    /// Check if the slice is valid
    ///
    void check() const
    {
        assert((_start > _end && _step <= 0) || (_start < _end && _step >= 0));
    }

    ///
    /// @return number of values in slice
    ///
    value_type size() const
    {
        if((_end <= _start && _step >= 0)
           || (_end >= _start && _step <= 0)) return 0;
        return (_end - _start + _step - (_step > 0 ? 1 : -1)) / _step;
    }

    ///
    /// @return ith element in slice
    /// does not check bounds, e.g. can return indices beyond end of slice
    ///
    value_type operator[](value_type i) const {
        return _start + i * _step;
    }

    ///
    /// A STL-like iterator for iterating through a slice.
    /// Supports pre-increment ++, unequality check != and dereference *
    ///
    struct iterator
    {
        iterator(const Slice * slice = nullptr)
            : slice_(slice), curr_(slice_ ? slice_->_start : 0)
        {}

        iterator& operator++() noexcept
        {
            curr_ += slice_->_step;
            if(curr_ >= slice_->_end) {
                *this = iterator();
            }
            return *this;
        }

        bool operator!=(const iterator & other) const noexcept
        {
            return slice_ != other.slice_ && curr_ != other.curr_;
        }

        Slice::value_type operator*() const noexcept
        {
            return curr_;
        }

    private:
        // we keep a reference to the orig slice
        const Slice * slice_;
        // and the current position/value
        Slice::value_type curr_;

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

    ///
    /// @return true if i is in slice
    ///
    bool covers(const value_type & i) const
    {
        assert(_step > 0 && _start >= 0);
        return (i >= _start
                && i < _end
                && ((i - (_start % _step)) % _step) == 0);
    }

    friend std::ostream &operator<<(std::ostream &output, const Slice & slc) {
        output << "(" << slc._start << ":" << slc._end << ":" << slc._step << ")";
        return output;
    }

    // Needed for serialization
    template<typename S>
    void serialize(S & ser)
    {
        ser.value8b(_start);
        ser.value8b(_end);
        ser.value8b(_step);
    }
};
