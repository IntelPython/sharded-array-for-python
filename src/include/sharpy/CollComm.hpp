// SPDX-License-Identifier: BSD-3-Clause

/*
    High-level collective communication functionality.
*/

#pragma once

#include "CppTypes.hpp"
#include "NDArray.hpp"
#include "PVSlice.hpp"

namespace SHARPY {

void gather_array(NDArray::ptr_type a_ptr, rank_type root, void *outPtr);

struct CollComm {
  using map_info_type = std::vector<std::vector<int>>;

  // Compute offset and displacements when mapping o_slc to n_slc. This is
  // necessary when slices are not equally partitioned. Basically we provide
  // the information how to ship all elements of a_ptr to where the equivalent
  // positions in b_ptr reside (using alltoall).  For example, if the 7th
  // element of n_slc resides on rank 2, element o_slc[7] will be shipped to
  // rank 2.
  //
  // Notice: slices might have strides > 1 and might not start at position 0.
  //         Results are provided relative to the given slices. It's up to the
  //         caller to translate to absolute positions.
  //
  // returns vector [counts_send, disp_send, counts_recv, disp_recv]
  static map_info_type map(const PVSlice &n_slc, const PVSlice &o_slc);

  template <typename T, typename U>
  static array_i::ptr_type coll_copy(std::shared_ptr<NDArray> b_ptr,
                                     const std::shared_ptr<NDArray> &a_ptr) {
#if 0
        assert(! a_ptr->is_sliced() && ! b_ptr->is_sliced());
        auto info = CollComm::map(b_ptr->slice(), a_ptr->slice());

        // Now we can send/recv directly to/from array buffers.
        getTransceiver()->alltoall(a_ptr->data(), // buffer_send
                                 info[0].data(),
                                 info[1].data(),
                                 DTYPE<T>::value,
                                 b_ptr->data(), // buffer_recv
                                 info[2].data(),
                                 info[3].data(),
                                 DTYPE<T>::value);
#endif

    return b_ptr;
  }

  template <typename T, typename U>
  static std::array<int, 4> coll_map(const std::shared_ptr<NDArray> &b_ptr,
                                     const std::shared_ptr<NDArray> &a_ptr,
                                     std::vector<U> &rbuff) {
#if 0
        auto info = CollComm::map(b_ptr->slice(), a_ptr->slice());

        auto nr = getTransceiver()->nranks();
        auto r = getTransceiver()->rank();

        // disable copy from local: first save local counts
        auto my_cnt_send = info[0][r];
        info[0][r] = 0;
        // auto my_dplc_send = info[1][r];
        auto my_cnt_recv = info[2][r];
        info[2][r] = 0;
        // auto my_dplc_send = info[3][r];
        // Now  adjust recv displacements.
        // We know data is ordered by ranks, so we can simply shift
        for(auto i=r+1; i<nr; ++i) {
            info[3][i] = info[3][i-1] + info[2][i-1];
        }
        // Create buffer. size is counts of all non-local elements
        rbuff.resize(info[3].back() + info[2].back());

        Buffer svec;
        const U * sbuff = nullptr;
        // if a_ptr is non-contiguous (strided) we need to first copy into buffer
        if(a_ptr->is_sliced()) {
            a_ptr->bufferize(NDSlice(a_ptr->slice().tile_shape()), svec);
            sbuff = reinterpret_cast<U*>(svec.data());
        } else sbuff = a_ptr->data();
        // Now we can send/recv directly to/from array buffers.
        getTransceiver()->alltoall(sbuff, // buffer_send
                                 info[0].data(),
                                 info[1].data(),
                                 DTYPE<U>::value,
                                 rbuff.data(), // buffer_recv
                                 info[2].data(),
                                 info[3].data(),
                                 DTYPE<U>::value);

        return {my_cnt_send, info[1][r], my_cnt_recv, info[3][r]};
#endif
    return {-1, -1, -1, -1};
  }

  template <typename A, typename B>
  static std::array<uint64_t, 2>
  coll_copy(const std::shared_ptr<NDArray> &a_ptr,
            const std::array<std::vector<NDSlice>, 2> &a_overlap,
            std::vector<B> &rbuff) {
#if 0
        if(a_overlap[0].empty()) return {0, 0};

        auto nr = getTransceiver()->nranks();
        auto rank = getTransceiver()->rank();
        int counts_send[nr] = {0};
        int disp_send[nr] = {0};
        int counts_recv[nr] = {0};
        int disp_recv[nr] = {0};
        Buffer sbuff;

        for(auto r=0; r<nr; ++r) {
            if(r) {
                disp_send[r] = disp_send[r-1] + counts_send[r-1];
                disp_recv[r] = disp_recv[r-1] + counts_recv[r-1];
            }
            if(r != rank) {
                counts_send[r] = a_overlap[0][r].size();
                a_ptr->bufferize(PVSlice(a_ptr->slice(), a_overlap[0][r]).tile_slice(), sbuff);
                counts_recv[r] = a_overlap[1][r].size();
            }
        }
        rbuff.resize((disp_recv[nr-1] + counts_recv[nr-1]));

        getTransceiver()->alltoall(sbuff.data(), // buffer_send
                                 &counts_send[0],
                                 &disp_send[0],
                                 DTYPE<A>::value,
                                 rbuff.data(), // buffer_recv
                                 &counts_recv[0],
                                 &disp_recv[0],
                                 DTYPE<B>::value);
        return {(uint64_t)disp_send[rank], (uint64_t)disp_recv[rank]};
#endif
    return {-1, -1};
  }
};
} // namespace SHARPY
