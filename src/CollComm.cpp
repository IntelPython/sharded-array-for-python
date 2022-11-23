// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/CollComm.hpp"

// Compute offset and displacements when mapping n_slc to o_slc. This is necessary when
// slices are not equally partitioned.
//
// We assume we split in first dimension.
// We also assume partitions are assigned to ranks in sequence from 0-N.
// With this we know that our buffers (old and new) get data in the
// same order. The only thing which might have changed is the tile-size.
// Actually, the tile-size might change only if old or new shape does not evenly
// distribute data (e.g. last partition is smaller).
// In theory we could re-shape in-place when the norm-tile-size does not change.
// This is not implemented: we need an extra mechanism to work with reshape-views or alike.
std::vector<std::vector<int>> CollComm::map(const PVSlice & n_slc, const PVSlice & o_slc)
{
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
