// SPDX-License-Identifier: BSD-3-Clause

#include <ddptensor/idtr.hpp>
#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/MPITransceiver.hpp>

#include <imex/Dialect/PTensor/IR/PTensorOps.h>

#include <cassert>
#include <memory>

using container_type = std::unordered_map<id_type, std::unique_ptr<DDPTensorImpl>>;

static container_type gtensors;
static id_type _nguid = -1;
inline id_type get_guid()
{
    return ++_nguid;
}

// Transceiver * theTransceiver = MPITransceiver();

template<typename T>
T * mr_to_ptr(void * ptr, intptr_t offset)
{
    auto mr = reinterpret_cast<intptr_t*>(ptr);
    return reinterpret_cast<T*>(ptr) + offset; // &mr.aligned[mr.offset]
}

extern "C" {

// Register a global tensor of given shape.
// Returns guid.
// The runtime does not own or manage any memory.
id_t idtr_init_dtensor(const uint64_t * shape, uint64_t nD)
{
    auto guid = get_guid();
    gtensors[guid] = std::unique_ptr<DDPTensorImpl>(nD ? new DDPTensorImpl(shape, nD) : new DDPTensorImpl);
    return guid;
}

id_t _idtr_init_dtensor(void * alloced, void * aligned, intptr_t offset, intptr_t size, intptr_t stride, uint64_t nD)
{
    return idtr_init_dtensor(mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the offsets (one for each dimension) of the local partition of a distributed tensor in number of elements.
// Result is stored in provided array.
void idtr_local_offsets(id_t guid, uint64_t * offsets, uint64_t nD)
{
    const auto & tnsr = gtensors.at(guid);
    auto slcs = tnsr->slice().local_slice().slices();
    assert(nD == slcs.size());
    int i = -1;
    for(auto s : slcs) {
        offsets[++i] = s._start;
    }
}

void _idtr_local_offsets(id_t guid, void * alloced, void * aligned, intptr_t offset, intptr_t size, intptr_t stride, uint64_t nD)
{
    idtr_local_offsets(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the shape (one size for each dimension) of the local partition of a distributed tensor in number of elements.
// Result is stored in provided array.
void idtr_local_shape(id_t guid, uint64_t * lshape, uint64_t N)
{
    const auto & tnsr = gtensors.at(guid);
    auto shp = tnsr->slice().local_slice().shape();
    std::copy(shp.begin(), shp.end(), lshape);
}

void _idtr_local_shape(id_t guid, void * alloced, void * aligned, intptr_t offset, intptr_t size, intptr_t stride, uint64_t nD)
{
    idtr_local_shape(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// convert id of our reduction op to id of imex::ptensor reduction op
static ReduceOpId mlir2ddpt(const ::imex::ptensor::ReduceOpId rop)
{
    switch(rop) {
    case ::imex::ptensor::MEAN:
        return MEAN;
    case ::imex::ptensor::PROD:
        return PROD;
    case ::imex::ptensor::SUM:
        return SUM;
    case ::imex::ptensor::STD:
        return STD;
    case ::imex::ptensor::VAR:
        return VAR;
    case ::imex::ptensor::MAX:
        return MAX;
    case MIN:
        return MIN;
    default:
        throw std::runtime_error("Unknown reduction operation");
    }
}

// Elementwise inplace allreduce
void idtr_reduce_all(void * inout, DTypeId dtype, uint64_t N, int op)
{

    getTransceiver()->reduce_all(inout, dtype, N, mlir2ddpt(static_cast<imex::ptensor::ReduceOpId>(op)));
}

// FIXME hard-coded 0d tensor
void _idtr_reduce_all(uint64_t * allocated, uint64_t * aligned, uint64_t offset, DTypeId dtype, int op)
{
    idtr_reduce_all(aligned + offset, dtype, 1, op);
}

} // extern "C"
