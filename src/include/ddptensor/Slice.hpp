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
  value_type start_, end_, step_;

  Slice()
    : start_(0), end_(-1), step_(1)
  {}

  ///
  /// Construct a slice/range [0:end:1[
  /// @param end first index out of range
  ///
  Slice(value_type end)
    : start_(0), end_(end), step_(1)
  {
  }

  ///
  /// Construct a slice [start:end:step[
  /// @param start first index of range
  /// @param end first index out of range
  /// @param step step between values in the slice
  ///
  Slice(value_type start, value_type end, value_type step=1)
    : start_(start), end_(end), step_(step)
  {
  }

  ///
  /// Construct a slice from a py::slice
  ///
  Slice(const py::slice & slc, int64_t len=std::numeric_limits<int64_t>::max())
  {
    ssize_t slc_len;
    slc.compute(len, &start_, &end_, &step_, &slc_len);
  }

  ///
  /// @return Subslice mapped into this' index space
  ///
  Slice slice(const Slice & slice) const
  {
    return {start_ + slice.start_ * step_, start_ + slice.end_ * step_, step_ * slice.step_};
  }

  ///
  /// @return py::slice representation
  ///
  py::slice pyslice() const
  {
    return py::slice(start_, end_, step_);
  }

  Slice normalize() const
  {
    return {0, size(), 1};
  }

  ///
  /// @return Slice with same shape, shifted left by off ignorig index space
  ///         and then normalized to stride 1
  /// Note: (end_-off) is the new zero before scaling down. start_ will also be scaled down by stride_
  ///
  Slice deflate(value_type off = 0) const
  {
    auto strt = start_ - off;
    return {strt/step_, strt + size(), 1};
    // if(step_ > 1) return {start_/step_, (end_+step_-1)/step_, 1};
    // return {start_, end_, step_};
  }

  ///
  /// @return Slice with same shape, shifted right by off and with stride strd
  ///
  Slice inflate(value_type off, value_type strd) const
  {
    auto strt = start_ + off;
    return {strt, strt + size() * strd, strd};
    // if(step_ > 1) return {start_*strd, (end_+step_-1)/step_, 1};
    // return {start_, end_, step_};
  }

  ///
  /// @return true if Slice is a lbock, e.g. step == 1
  ///
  bool is_block() const
  {
    return step_ == 1;
  }

  ///
  /// Check if the slice is valid
  ///
  void check() const
  {
    assert((start_ > end_ && step_ <= 0) || (start_ < end_ && step_ >= 0));
  }

  ///
  /// @return number of values in slice
  ///
  value_type size() const
  {
    if((end_ <= start_ && step_ >= 0)
       || (end_ >= start_ && step_ <= 0)) return 0;
    return (end_ - start_ + step_ - (step_ > 0 ? 1 : -1)) / step_;
  }

  ///
  /// @return ith element in slice
  /// does not check bounds, e.g. can return indices beyond end of slice
  ///
  value_type operator[](value_type i) const {
    return start_ + i * step_;
  }

  ///
  /// A STL-like iterator for iterating through a slice.
  /// Supports pre-increment ++, unequality check != and dereference *
  ///
  struct iterator
  {
    iterator(const Slice * slice = nullptr)
      : slice_(slice), curr_(slice_ ? slice_->start_ : 0)
    {}

    iterator& operator++() noexcept
    {
      curr_ += slice_->step_;
      if(curr_ >= slice_->end_) {
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
    assert(step_ > 0 && start_ >= 0);
    // std::cerr << "\t\t\t" << i << " " << start_ << " " << end_ << " " << step_ << " " << ((i - (start_ % step_)) % step_) << std::endl;
    return (i >= start_
            && i < end_
            && ((i - (start_ % step_)) % step_) == 0);
  }

  ///
  /// @return copy of this Slice, trimmed to given start and end.
  /// @param s Start index to trim to
  /// @param e End index to trim to
  ///
  Slice trim(value_type s, value_type e, value_type shift = 0) const
  {
    assert(step_ > 0);
    auto start = start_;
    if(s > start_) {
      auto m = (s - start) % step_;
      if(m) start = s + step_ - m;
      else start = s;
    }
    auto end = std::min(e, end_);
    return {start-shift, end-shift, step_};
  }

  Slice map(const Slice & slc) const
  {
    value_type diff = slc.start_ - start_;
    value_type start = diff / step_;
    assert(diff % step_ == 0);
    assert((slc.step_ == step_));
    return {start, start + slc.size(), 1};
  }

  ///
  /// @return copy of this Slice, shifted left by given amount.
  ///
  Slice shift(value_type s) const
  {
    return {start_ - s, end_ - s, step_};
  }

  friend std::ostream &operator<<(std::ostream &output, const Slice & slc) {
    output << "(" << slc.start_ << ":" << slc.end_ << ":" << slc.step_ << ")";
    return output;
  }

  // Needed for serialization
  template<typename S>
  void serialize(S & ser)
  {
    ser.value8b(start_);
    ser.value8b(end_);
    ser.value8b(step_);
  }
};
