// SPDX-License-Identifier: BSD-3-Clause

#include <ddptensor/idtr.hpp>
// #include <ddptensor/jit/mlir.hpp>
#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/MPITransceiver.hpp>

#include <imex/Dialect/PTensor/IR/PTensorDefs.h>

#include <cassert>
#include <memory>
#include <iostream>

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

// Return number of ranks/processes in given team/communicator
uint64_t idtr_nprocs(int64_t team)
{
    return getTransceiver()->nranks();
}
#pragma weak _idtr_nprocs = idtr_nprocs

// Return rank in given team/communicator
uint64_t idtr_prank(int64_t team)
{
    return getTransceiver()->rank();
}
#pragma weak _idtr_prank = idtr_prank

// Register a global tensor of given shape.
// Returns guid.
// The runtime does not own or manage any memory.
id_t idtr_init_dtensor(const uint64_t * shape, uint64_t nD)
{
    auto guid = get_guid();
    // gtensors[guid] = std::unique_ptr<DDPTensorImpl>(nD ? new DDPTensorImpl(shape, nD) : new DDPTensorImpl);
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
#if 0
    const auto & tnsr = gtensors.at(guid);
    auto slcs = tnsr->slice().local_slice().slices();
    assert(nD == slcs.size());
    int i = -1;
    for(auto s : slcs) {
        offsets[++i] = s._start;
    }
#endif
}

void _idtr_local_offsets(id_t guid, void * alloced, void * aligned, intptr_t offset, intptr_t size, intptr_t stride, uint64_t nD)
{
    idtr_local_offsets(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the shape (one size for each dimension) of the local partition of a distributed tensor in number of elements.
// Result is stored in provided array.
void idtr_local_shape(id_t guid, uint64_t * lshape, uint64_t N)
{
#if 0
    const auto & tnsr = gtensors.at(guid);
    auto shp = tnsr->slice().local_slice().shape();
    std::copy(shp.begin(), shp.end(), lshape);
#endif
}

void _idtr_local_shape(id_t guid, void * alloced, void * aligned, intptr_t offset, intptr_t size, intptr_t stride, uint64_t nD)
{
    idtr_local_shape(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}
} // extern "C"

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

static DTypeId mlir2ddpt(const ::imex::ptensor::DType dt)
{
    switch(dt) {
    case ::imex::ptensor::DType::F64:
        return FLOAT64;
        break;
    case ::imex::ptensor::DType::I64:
        return INT64;
        break;
    case ::imex::ptensor::DType::U64:
        return UINT64;
        break;
    case ::imex::ptensor::DType::F32:
        return FLOAT32;
        break;
    case ::imex::ptensor::DType::I32:
        return INT32;
        break;
    case ::imex::ptensor::DType::U32:
        return UINT32;
        break;
    case ::imex::ptensor::DType::I16:
        return INT16;
        break;
    case ::imex::ptensor::DType::U16:
        return UINT16;
        break;
    case ::imex::ptensor::DType::I8:
        return INT8;
        break;
    case ::imex::ptensor::DType::U8:
        return UINT8;
        break;
    case ::imex::ptensor::DType::I1:
        return BOOL;
        break;
    default:
        throw std::runtime_error("unknown dtype");
    };
}


template<typename T, typename OP>
void forall(uint64_t d, const T * cptr, const int64_t * sizes, const int64_t * strides, uint64_t nd, OP op)
{
    auto stride = strides[d];
    auto sz = sizes[d];
    if(d==nd-1) {
        for(auto i=0; i<sz; ++i) {
            op(&cptr[i*stride]);
        }
    } else {
        for(auto i=0; i<sz; ++i) {
            forall(d+1, cptr, sizes, strides, nd, op);
        }
    }
}

bool is_contiguous(const int64_t * sizes, const int64_t * strides, uint64_t nd)
{
    if(nd == 0) return true;
    if(strides[nd-1] != 1) return false;
    auto sz = 1;
    for(auto i=nd-1; i>0; --i) {
        sz *= sizes[i];
        if(strides[i-1] != sz) return false;
    }
    return true;
}

void * bufferize(void * cptr, DTypeId dtype, const int64_t * sizes, const int64_t * strides, uint64_t nd, void * out)
{
    if(is_contiguous(sizes, strides, nd)) {
        return cptr;
    } else {
        dispatch(dtype, cptr, [sizes, strides, nd, out](auto * ptr) {
            auto buff = static_cast<decltype(ptr)>(out);
            forall(0, ptr, sizes, strides, nd, [&buff](const auto * in) {
                *buff = *in;
                ++buff;
            });
        });
        return out;
    }
}

extern "C" {
// Elementwise inplace allreduce
void idtr_reduce_all(void * inout, DTypeId dtype, uint64_t N, ReduceOpId op)
{
    getTransceiver()->reduce_all(inout, dtype, N, op);
}

// FIXME hard-coded for contiguous layout
void _idtr_reduce_all(uint64_t rank, void * data, const int64_t * sizes, const int64_t * strides, int dtype, int op)
{
    assert(rank == 0 || strides[rank-1] == 1);
    idtr_reduce_all(data,
                    mlir2ddpt(static_cast<::imex::ptensor::DType>(dtype)),
                    rank ? rank : 1,
                    mlir2ddpt(static_cast<imex::ptensor::ReduceOpId>(op)));
}

void _idtr_rebalance(uint64_t rank, const int64_t * gShape, const int64_t * lOffs,
                     void * data, const int64_t * sizes, const int64_t * strides, int dtype,
                     uint64_t outRank, void * out, const int64_t * outSizes, const int64_t * outStrides)
{
    assert(rank);
    is_contiguous(outSizes, outStrides, outRank);
    auto N = (int64_t)getTransceiver()->nranks();
    auto myOff = lOffs[0];
    auto mySz = sizes[0];
    auto myEnd = myOff + mySz;
    auto tSz = gShape[0];
    auto sz = (tSz + N - 1) / N;
    auto ddpttype = mlir2ddpt(static_cast<::imex::ptensor::DType>(dtype));
    auto nSz = std::accumulate(&sizes[1], &sizes[rank], 1, std::multiplies<int64_t>());
    std::vector<int> soffs(N);
    std::vector<int> sszs(N, 0);
    for(auto i=0; i<N; ++i) {
        auto tOff = i * sz;
        auto tEnd = std::min(tSz, tOff + sz);
        if(tEnd > myOff && tOff < myEnd) {
            // We have a target partition which is inside my local data
            // we now compute what data goes to this target partition
            auto start = std::max(myOff, tOff);
            auto end = std::min(myEnd, tEnd);
            soffs[i] = (int)(start - myOff) * nSz;
            sszs[i] = (int)(end - start) * nSz;
        } else {
            soffs[i] = i ? soffs[i-1] + sszs[i-1] : 0;
        }
    }
    // we now send our send sizes to others and receiver theirs
    std::vector<int> rszs(N);
    getTransceiver()->alltoall(sszs.data(), 1, INT32, rszs.data());
    // For the actual alltoall we need the receive-displacements
    std::vector<int> roffs(N);
    roffs[0] = 0;
    for(auto i=1; i<N; ++i) {
        // compute for all i > 0
        roffs[i] = roffs[i-1] + rszs[i-1];
    }
    // create send buffer (might be strided!)
    Buffer buff(nSz * mySz * sizeof_dtype(ddpttype));
    auto ptr = bufferize(data, ddpttype, sizes, strides, rank, buff.data());
    // Finally communicate elements
    getTransceiver()->alltoall(ptr, sszs.data(), soffs.data(), ddpttype, out, rszs.data(), roffs.data());
}
} // extern "C"
