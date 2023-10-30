// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/CollComm.hpp"

namespace DDPT {

void bufferize(DDPTensorImpl::ptr_type a_ptr, void *outPtr) {
  dispatch(a_ptr->dtype(), a_ptr->data(), [&a_ptr, outPtr](auto *ptr) {
    auto buff = static_cast<decltype(ptr)>(outPtr);
    auto shp = a_ptr->local_shape();
    if (shp) {
      forall(0, ptr, shp, a_ptr->local_strides(), a_ptr->ndims(),
             [&buff](const auto *in) {
               *buff = *in;
               ++buff;
             });
    } else {
      buff[0] = ptr[0];
    }
  });
}

// @param outPtr touched only if on root and/or root==REPLICATED or if not
// distributed
void gather_tensor(DDPTensorImpl::ptr_type a_ptr, rank_type root,
                   void *outPtr) {
  auto trscvr = a_ptr->transceiver();

  if (!trscvr || a_ptr->owner() == REPLICATED) {
    bufferize(a_ptr, outPtr);
    return;
  }

  auto nranks = trscvr->nranks();
  auto myrank = trscvr->rank();
  bool sendonly = root != REPLICATED && root != myrank;

  auto dtype = a_ptr->dtype();
  auto mysizes = a_ptr->local_shape();
  auto mysz = mysizes[0];
  auto myoff = a_ptr->local_offsets()[0];
  auto nd = a_ptr->ndims();
  auto gshape = a_ptr->shape();
  auto myTileSz =
      std::accumulate(&gshape[1], &gshape[nd], 1, std::multiplies<int64_t>());

  // allgather process local offset and sizes
  std::vector<int> displacements(nranks);
  std::vector<int> counts(nranks, 2);
  std::vector<int> szsAndOffs(2 * nranks);
  for (size_t i = 0; i < nranks; ++i) {
    displacements[i] = i * 2;
  }
  szsAndOffs[2 * myrank + 0] = myoff; // FIXME split dim
  szsAndOffs[2 * myrank + 1] = mysz;
  trscvr->gather(szsAndOffs.data(), counts.data(), displacements.data(), INT32,
                 REPLICATED);

  // compute each pranks local contribution
  int64_t curr = 0;
  for (auto i = 0; i < nranks; ++i) {
    assert(szsAndOffs[i * 2] * myTileSz == curr);
    displacements[i] = curr;
    counts[i] = szsAndOffs[i * 2 + 1] * myTileSz;
    curr += counts[i];
  }

  // create buffer/numpy array and copy
  void *ptr = nullptr;
  bool need_del = false;
  if (sendonly) {
    if (mysz > 0 && a_ptr->is_sliced()) {
      ptr = new char[mysz * sizeof_dtype(dtype) * myTileSz];
      bufferize(a_ptr, ptr);
      need_del = true;
    } else if (mysz > 0) {
      ptr = a_ptr->data();
    }
  } else {
    ptr = outPtr;
    bufferize(a_ptr, &(static_cast<char *>(
                         ptr)[displacements[myrank] * sizeof_dtype(dtype)]));
  }

  // final gather
  trscvr->gather(ptr, counts.data(), displacements.data(), dtype, root);

  if (need_del)
    delete[] static_cast<char *>(ptr);
}

// Compute offset and displacements when mapping n_slc to o_slc. This is
// necessary when slices are not equally partitioned.
//
// We assume we split in first dimension.
// We also assume partitions are assigned to ranks in sequence from 0-N.
// With this we know that our buffers (old and new) get data in the
// same order. The only thing which might have changed is the tile-size.
// Actually, the tile-size might change only if old or new shape does not evenly
// distribute data (e.g. last partition is smaller).
// In theory we could re-shape in-place when the norm-tile-size does not change.
// This is not implemented: we need an extra mechanism to work with
// reshape-views or alike.
std::vector<std::vector<int>> CollComm::map(const PVSlice &n_slc,
                                            const PVSlice &o_slc) {
#if 0
    auto nr = getTransceiver()->nranks();
    std::vector<int> counts_send(nr, 0);
    std::vector<int> disp_send(nr, 0);
    std::vector<int> counts_recv(nr, 0);
    std::vector<int> disp_recv(nr, 0);

    // norm tile-size of  orig array
    auto o_ntsz =  o_slc.tile_size(0);
    // tilesize of my local partition of orig array
    auto o_tsz = o_slc.tile_size();
    // linearized local slice of orig array
    auto o_llslc = Slice(o_ntsz * getTransceiver()->rank(), o_ntsz * getTransceiver()->rank() + o_tsz);

    // norm tile-size of new (reshaped) array
    auto n_ntsz =  n_slc.tile_size(0);
    // tilesize of my local partition of new (reshaped) array
    auto n_tsz = n_slc.tile_size();
    // linearized/flattened/1d local slice of new (reshaped) array
    auto n_llslc = Slice(n_ntsz * getTransceiver()->rank(), n_ntsz * getTransceiver()->rank() + n_tsz);

    for(auto r=0; r<nr; ++r) {
        // determine what I receive from rank r
        // e.g. which parts of my new slice overlap with rank r's old slice
        // Get local slice of rank r of orig array
        auto o_rslc = o_slc.tile_slice(r);
        // Flatten to 1d
        auto o_lrslc = Slice(o_ntsz * r, o_ntsz * r + o_rslc.size());
        // Determine overlap with local partition of linearized new array
        auto roverlap = n_llslc.overlap(o_lrslc);
        // number of elements to be received from rank r
        counts_recv[r] = roverlap.size();
        // displacement in new array where elements from rank r get copied to
        disp_recv[r] = roverlap._start - n_llslc._start;

        // determine what I send to rank r
        // e.g. which parts of my old slice overlap with rank r's new slice
        // Get local slice of rank r of new array
        auto n_rslc = n_slc.tile_slice(r);
        // Flatten to 1d
        auto n_lrslc = Slice(n_ntsz * r, n_ntsz * r + n_rslc.size());
        // Determine overlap with local partition of linearized orig array
        auto soverlap = o_llslc.overlap(n_lrslc);
        // number of elements to be send to rank r
        counts_send[r] = soverlap.size();
        // displacement in orig array where elements from rank r get copied from
        disp_send[r] = soverlap._start - o_llslc._start;
    }
    return {counts_send, disp_send, counts_recv, disp_recv};
#endif // if 0
  return {};
}
} // namespace DDPT
