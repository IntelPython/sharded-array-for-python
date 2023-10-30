// SPDX-License-Identifier: BSD-3-Clause

// deprecated

#include "ddptensor/PVSlice.hpp"

namespace DDPT {

using offsets_type = std::vector<uint64_t>;

BasePVSlice::BasePVSlice(const shape_type &shape, int split)
    : _offset(split == NOSPLIT
                  ? 0
                  : (shape[split] + getTransceiver()->nranks() - 1) /
                        getTransceiver()->nranks()),
      _shape(shape), _split_dim(split) {
  _tile_size = VPROD(_shape) / shape[_split_dim] * _offset;
}

BasePVSlice::BasePVSlice(shape_type &&shape, int split)
    : _offset(split == NOSPLIT
                  ? 0
                  : (shape[split] + getTransceiver()->nranks() - 1) /
                        getTransceiver()->nranks()),
      _shape(std::move(shape)), _split_dim(split) {
  if (_split_dim == NOSPLIT)
    _tile_size = VPROD(_shape);
  else
    _tile_size =
        _shape.size() ? VPROD(_shape) / _shape[_split_dim] * _offset : 1;
}

bool BasePVSlice::is_equally_tiled() const {
  return _shape[_split_dim] == getTransceiver()->nranks() * _offset;
}

uint64_t BasePVSlice::offset() const { return _offset; }

uint64_t BasePVSlice::tile_size(rank_type rank) const {
  // only rank 0 is guaranteed to have _tile_size, all other parts can be <
  // _tile_size
  if (rank == 0)
    return _tile_size;
  auto sz = VPROD(_shape);
  auto off = rank * _tile_size;
  if (sz >= off)
    return _tile_size;
  auto r = off - sz;
  // if r < _tile_size it's the remainder, otherwise we are past the end of the
  // global array
  return r < _tile_size ? r : 0UL;
}

shape_type BasePVSlice::tile_shape(rank_type rank) const {
  // only rank 0 is guaranteed to have _tile_size, all other parts can be <
  // _tile_size
  shape_type r(_shape);
  if (rank == 0)
    r[_split_dim] = offset();
  else {
    auto end = (rank + 1) * offset();
    if (r[_split_dim] >= end)
      r[_split_dim] = offset();
    else {
      auto diff = end - r[_split_dim];
      // if diff < offset() it's the remainder, otherwise we are past the end of
      // the global array
      r[_split_dim] = diff < offset() ? offset() - diff : 0UL;
    }
  }
  return r;
}

int BasePVSlice::split_dim() const { return _split_dim; }

const shape_type &BasePVSlice::shape() const { return _shape; }

rank_type BasePVSlice::owner(const NDSlice &slice) const {
  return split_dim() == NOSPLIT ? getTransceiver()->rank()
                                : slice.dim(split_dim())._start / offset();
}

PVSlice::PVSlice(const shape_type &shp, int split)
    : _slice(shp), _base(std::make_shared<BasePVSlice>(shp, split)), _shape() {}

PVSlice::PVSlice(shape_type &&shp, int split)
    : _slice(shp), // std::move in next line
      _base(std::make_shared<BasePVSlice>(std::move(shp), split)), _shape() {}

PVSlice::PVSlice(const shape_type &shp, const NDSlice &slc, int split)
    : _slice(slc), _base(std::make_shared<BasePVSlice>(shp, split)), _shape() {}

PVSlice::PVSlice(shape_type &&shp, NDSlice &&slc, int split)
    : _slice(std::move(slc)),
      _base(std::make_shared<BasePVSlice>(std::move(shp), split)), _shape() {}

PVSlice::PVSlice(const PVSlice &org, const NDSlice &slice)
    : _slice(std::move(org._slice.slice(slice))), _base(org._base), _shape() {}

PVSlice::PVSlice(const PVSlice &org, rank_type rank)
    : _slice(std::move(org.local_slice(rank))), _base(org._base), _shape() {}

PVSlice::PVSlice(BasePVSlicePtr bp, const NDSlice &slice)
    : _slice(slice), _base(bp), _shape() {}

uint64_t PVSlice::ndims() const { return _slice.ndims(); }

int PVSlice::split_dim() const { return _base->split_dim(); }

bool PVSlice::is_sliced() const { return base_shape() != shape(); }

bool PVSlice::local_is_contiguous(rank_type rank) const {
  assert(split_dim() == 0);
  auto tshp = tile_shape(rank);
  auto tslc = tile_slice(rank);
  for (auto i = 0; i < ndims(); ++i) {
    auto slci = tslc.dim(i);
    if (slci._step != 1 || (slci._start > 0 && i > 0) ||
        (slci._end < tshp[i] && i > 0))
      return false;
  }
  return true;
}

bool PVSlice::is_equally_tiled() const { return _base->is_equally_tiled(); }

const NDSlice &PVSlice::slice() const { return _slice; }

const shape_type &PVSlice::shape() const {
  if (_shape.size() != _slice.ndims()) {
    _shape.resize(_slice.ndims());
    int i = -1;
    for (auto &shp : _shape)
      shp = _slice.dim(++i).size();
  }
  return _shape;
}

uint64_t PVSlice::size() const { return slice().size(); }

uint64_t PVSlice::tile_size(rank_type rank) const {
  return _base->tile_size(rank);
}

shape_type PVSlice::tile_shape(rank_type rank) const {
  return _base->tile_shape(rank);
}

NDSlice PVSlice::tile_slice(rank_type rank) const {
  if (_base->split_dim() == NOSPLIT) {
    return rank == getTransceiver()->rank() ? slice() : NDSlice();
  }
  return _slice.trim_shift(_base->split_dim(), rank * _base->offset(),
                           (rank + 1) * _base->offset(),
                           rank * _base->offset());
}

NDSlice PVSlice::local_slice(rank_type rank) const {
  if (_base->split_dim() == NOSPLIT) {
    return rank == getTransceiver()->rank() ? slice() : NDSlice();
  }
  return _slice.trim(_base->split_dim(), rank * _base->offset(),
                     (rank + 1) * _base->offset());
}

shape_type PVSlice::local_shape(rank_type rank) const {
  return local_slice(rank).shape();
}

uint64_t PVSlice::local_size(rank_type rank) const {
  return local_slice(rank).size();
}

const shape_type &PVSlice::base_shape() const { return _base->shape(); }

rank_type PVSlice::owner(const NDSlice &slice) const {
  return _base->owner(slice);
}

#if 0
NDSlice PVSlice::normalized_slice() const
{
    return _slice.normalize(_base->split_dim());
}
#endif

NDSlice PVSlice::map_slice(const NDSlice &slc) const { return _slice.map(slc); }

std::array<std::vector<NDSlice>, 2>
PVSlice::map_ranks(const PVSlice &o_slc) const {
  auto o_sz = o_slc.size();
  auto d_sz = size();
  if (d_sz <= 1 && o_sz > 1)
    throw std::runtime_error("Cannot map nd-tensor to scalar/0d-tensor.");
  if (o_sz <= 1)
    return {};

  auto nr = getTransceiver()->nranks();
  std::vector<NDSlice> sends(nr);
  std::vector<NDSlice> recvs(nr);

  // rank's slice of origin, relative to o_slc
  auto my_o_slc = o_slc.map_slice(o_slc.local_slice());
  // rank's slice of destination, relative to d_slc
  auto my_d_slc = this->map_slice(this->local_slice());

  for (auto r = 0; r < nr; ++r) {
    // determine what I receive from rank r
    // e.g. which parts of my destination slice overlap with rank r's origin
    // slice Get local slice of rank r of origin array
    auto r_o_slc = o_slc.map_slice(o_slc.local_slice(r));
    // Determine overlap with my destination
    auto roverlap = my_d_slc.overlap(r_o_slc);
    // push to result
    recvs[r] = std::move(roverlap);

    // determine what I send to rank r
    // e.g. which parts of my origin slice overlap with rank r's destination
    // slice Get local slice of rank r of destination array
    auto r_d_slc = this->map_slice(this->local_slice(r));
    // Determine overlap with my destination
    auto soverlap = my_o_slc.overlap(r_d_slc);
    // push to result
    sends[r] = std::move(soverlap);
  }
  return {sends, recvs};
}

bool PVSlice::need_reduce(const dim_vec_type &dims) const {
  if (_base->split_dim() == NOSPLIT)
    return false;
  auto nd = dims.size();
  // Reducing to a single scalar or over a subset of dimensions *including* the
  // split axis.
  if (nd == 0 || nd == _slice.ndims() ||
      std::find(dims.begin(), dims.end(), _base->split_dim()) != dims.end())
    return true;

  // *not* reducing over split axis
  return false;
} ///

std::ostream &operator<<(std::ostream &output, const PVSlice &slc) {
  output << "{slice=" << slc.slice() << "base=" << to_string(slc._base->shape())
         << "offset=" << slc._base->offset() << "}";
  return output;
}
} // namespace DDPT
