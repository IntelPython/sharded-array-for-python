// SPDX-License-Identifier: BSD-3-Clause

#include <ddptensor/idtr.hpp>
#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/MPITransceiver.hpp>

#include <cassert>
#include <memory>

using container_type = std::unordered_map<id_type, std::unique_ptr<DDPTensorImpl>>;

static container_type gtensors;

// Transceiver * theTransceiver = MPITransceiver();

extern "C" {

// Register a global tensor of given shape.
// Accepts a guid which might have been reserved before. Returns guid (reserved or new).
// The runtime does not own or manage any memory.
id_t idtr_init_dtensor(const uint64_t * shape, uint64_t N, id_t guid)
{
    assert(guid != UNKNOWN_GUID);
    gtensors[guid] = std::unique_ptr<DDPTensorImpl>(new DDPTensorImpl(shape, N));
    return guid;
}

// Get the offsets (one for each dimension) of the local partition of a distributed tensor in number of elements.
// Result is stored in provided array.
void idtr_local_offsets(id_t guid, uint64_t * offsets, uint64_t N)
{
    const auto & tnsr = gtensors.at(guid);
    auto slcs = tnsr->slice().local_slice().slices();
    int i = -1;
    for(auto s : slcs) {
        offsets[++i] = s._start;
    }
}

// Get the shape (one size for each dimension) of the local partition of a distributed tensor in number of elements.
// Result is stored in provided array.
void idtr_local_shape(id_t guid, uint64_t * lshape, uint64_t N)
{
    const auto & tnsr = gtensors.at(guid);
    auto shp = tnsr->slice().local_slice().shape();
    std::copy(shp.begin(), shp.end(), lshape);
}

// Elementwise inplace allreduce
void idtr_reduce_all(void * inout, DTypeId dtype, size_t N, RedOpType op)
{
    getTransceiver()->reduce_all(inout, dtype, N, op);
}

} // extern "C"
