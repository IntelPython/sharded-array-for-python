// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ddptensor/CppTypes.hpp>

extern "C" {

    constexpr id_t UNKNOWN_GUID = -1;

    // Register a global tensor of given shape.
    // Accepts a guid which might have been reserved before. Returns guid (reserved or new).
    // The runtime does not own or manage any memory.
    id_t idtr_nit_dtensor(const uint64_t * shape, uint64_t N, id_t guid = UNKNOWN_GUID);

    // Get the offsets (one for each dimension) of the local partition of a distributed tensor in number of elements.
    // Result is stored in provided array.
    void idtr_local_offsets(id_t guid, uint64_t * offsets, uint64_t N);

    // Get the shape (one size for each dimension) of the local partition of a distributed tensor in number of elements.
    // Result is stored in provided array.
    void idtr_local_shape(id_t guid, uint64_t * lshape, uint64_t N);

    // Elementwise inplace allreduce
    void idtr_reduce_all(void * inout, DTypeId T, size_t N, RedOpType op);

} // extern "C"
