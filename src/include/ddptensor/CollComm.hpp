// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "x.hpp"

struct CollComm
{
    // We assume we split in first dimension.
    // We also assume partitions are assigned to ranks in sequence from 0-N.
    // With this we know that our buffers (old and new) get data in the
    // same order. The only thing which might have changed is the tile-size.
    // Actually, the tile-size might change only if old or new shape does not evenly
    // distribute data (e.g. last partition is smaller).
    // In theory we could re-shape in-place when the norm-tile-size does not change.
    // This is not implemented: we need an extra mechanism to work with reshape-views or alike.
    template<typename T, typename U>
    static tensor_i::ptr_type coll_copy(std::shared_ptr<x::DPTensorX<T>> b_ptr, const std::shared_ptr<x::DPTensorX<U>> & a_ptr)
    {
        assert(! a_ptr->is_sliced() && ! b_ptr->is_sliced());

        auto o_slc = a_ptr->slice();
        // norm tile-size of  orig array
        auto o_ntsz =  o_slc.tile_size(0);
        // tilesize of my local partition of orig array
        auto o_tsz = o_slc.tile_size();
        // linearized local slice of orig array
        auto o_llslc = Slice(o_ntsz * theTransceiver->rank(), o_ntsz * theTransceiver->rank() + o_tsz);
            
        PVSlice n_slc = b_ptr->slice();
        // norm tile-size of new (reshaped) array
        auto n_ntsz =  n_slc.tile_size(0);
        // tilesize of my local partition of new (reshaped) array
        auto n_tsz = n_slc.tile_size();
        // linearized/flattened/1d local slice of new (reshaped) array
        auto n_llslc = Slice(n_ntsz * theTransceiver->rank(), n_ntsz * theTransceiver->rank() + n_tsz);
            
        auto nr = theTransceiver->nranks();
        // We need a few C-arrays for MPI (counts and displacements in send/recv buffers)
        int counts_send[nr] = {0};
        int disp_send[nr] = {0};
        int counts_recv[nr] = {0};
        int disp_recv[nr] = {0};

        for(auto r=0; r<nr; ++r) {
            // determine what I receive from rank r
            // e.g. which parts of my new slice overlap with rank r's old slice
            // Get local slice of rank r of orig array
            auto o_rslc = o_slc.local_slice_of_rank(r);
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
            auto n_rslc = n_slc.local_slice_of_rank(r);
            // Flatten to 1d
            auto n_lrslc = Slice(n_ntsz * r, n_ntsz * r + n_rslc.size());
            // Determine overlap with local partition of linearized orig array
            auto soverlap = o_llslc.overlap(n_lrslc);
            // number of elements to be send to rank r
            counts_send[r] = soverlap.size();
            // displacement in orig array where elements from rank r get copied from
            disp_send[r] = soverlap._start - o_llslc._start;
        }

        // Now we can send/recv directly to/from xarray buffers.
        theTransceiver->alltoall(a_ptr->xarray().data(), // buffer_send
                                 counts_send,
                                 disp_send,
                                 DTYPE<T>::value,
                                 b_ptr->xarray().data(), // buffer_recv
                                 counts_recv,
                                 disp_recv,
                                 DTYPE<T>::value);
            
        return b_ptr;
    }
};
