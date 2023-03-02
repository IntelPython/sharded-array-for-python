// SPDX-License-Identifier: BSD-3-Clause

/*
    Partitioning feature. Relict.
*/

#pragma once

class Chunker {
public:
  enum : int { PTR = 0, SLC = 1 };
  using chunk = std::tuple<const void *, NDSlice>;
  using chunk_vec = std::vector<chunk>;

private:
  chunk_vec _chunks;

public:
  Chunker(chunk_vec &&cv) : _chunks(std::move(cv)) {}

  class Copier {
    const chunk_vec *_chunks;
    chunk_vec::const_iterator _curr_chunk;
    NDSlice::iterator _curr_pos;

  public:
    Copier() : _chunks(0), _curr_chunk(0), _curr_pos() {}
    Copier(const chunk_vec *chunks)
        : _chunks(chunks), _curr_chunk(_chunks->begin()),
          _curr_pos(_chunks->size() ? std::get<SLC>((*_curr_chunk)).begin()
                                    : NDSlice::iterator()) {}

    bool operator!=(const Copier &other) const noexcept {
      return _chunks != other._chunks || _curr_pos != other._curr_pos ||
             _curr_chunk != other._curr_chunk;
    }

    bool operator==(const Copier &other) const noexcept {
      return !operator!=(other);
    }

    template <typename F, typename T>
    Copier copy(T *buff, uint64_t capacity, uint64_t &ncopied) noexcept {
      if (*this == Copier())
        return *this;
      ncopied = 0;
      while (ncopied < capacity) {
        while (_curr_pos == std::get<SLC>(*_curr_chunk).end()) {
          ++_curr_chunk;
          if (_curr_chunk == _chunks->end()) {
            *this = Copier();
            return *this;
          }
          _curr_pos = std::get<SLC>(*_curr_chunk).begin();
        }
        uint64_t _nc;
        _curr_pos = _curr_pos.fill_buffer(
            buff + ncopied, capacity - ncopied, _nc,
            reinterpret_cast<const F *>(std::get<PTR>(*_curr_chunk)));
        ncopied += _nc;
      }
      return *this;
    }
  };

  Copier copier() const { return Copier(&_chunks); }

  Copier end() const { return Copier(); }
};
