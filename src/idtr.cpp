// SPDX-License-Identifier: BSD-3-Clause

/*
    Intel Distributed Runtime for MLIR
*/

#include <ddptensor/idtr.hpp>
// #include <ddptensor/jit/mlir.hpp>
#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/MPITransceiver.hpp>

#include <imex/Dialect/PTensor/IR/PTensorDefs.h>

#include <cassert>
#include <iostream>
#include <memory>

using container_type =
    std::unordered_map<id_type, std::unique_ptr<DDPTensorImpl>>;

static container_type gtensors;
static id_type _nguid = -1;
inline id_type get_guid() { return ++_nguid; }

// Transceiver * theTransceiver = MPITransceiver();

template <typename T> T *mr_to_ptr(void *ptr, intptr_t offset) {
  auto mr = reinterpret_cast<intptr_t *>(ptr);
  return reinterpret_cast<T *>(ptr) + offset; // &mr.aligned[mr.offset]
}

extern "C" {

#define NO_TRANSCEIVER
#ifdef NO_TRANSCEIVER
static void initMPIRuntime() {
  if (getTransceiver() == nullptr)
    init_transceiver(new MPITransceiver(false));
}
#endif

// Return number of ranks/processes in given team/communicator
uint64_t idtr_nprocs(Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif
  return tc->nranks();
}
#pragma weak _idtr_nprocs = idtr_nprocs

// Return rank in given team/communicator
uint64_t idtr_prank(Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif
  return tc->rank();
}
#pragma weak _idtr_prank = idtr_prank

// Register a global tensor of given shape.
// Returns guid.
// The runtime does not own or manage any memory.
id_t idtr_init_dtensor(const uint64_t *shape, uint64_t nD) {
  auto guid = get_guid();
  // gtensors[guid] = std::unique_ptr<DDPTensorImpl>(nD ? new
  // DDPTensorImpl(shape, nD) : new DDPTensorImpl);
  return guid;
}

id_t _idtr_init_dtensor(void *alloced, void *aligned, intptr_t offset,
                        intptr_t size, intptr_t stride, uint64_t nD) {
  return idtr_init_dtensor(mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the offsets (one for each dimension) of the local partition of a
// distributed tensor in number of elements. Result is stored in provided array.
void idtr_local_offsets(id_t guid, uint64_t *offsets, uint64_t nD) {
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

void _idtr_local_offsets(id_t guid, void *alloced, void *aligned,
                         intptr_t offset, intptr_t size, intptr_t stride,
                         uint64_t nD) {
  idtr_local_offsets(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the shape (one size for each dimension) of the local partition of a
// distributed tensor in number of elements. Result is stored in provided array.
void idtr_local_shape(id_t guid, uint64_t *lshape, uint64_t N) {
#if 0
    const auto & tnsr = gtensors.at(guid);
    auto shp = tnsr->slice().local_slice().shape();
    std::copy(shp.begin(), shp.end(), lshape);
#endif
}

void _idtr_local_shape(id_t guid, void *alloced, void *aligned, intptr_t offset,
                       intptr_t size, intptr_t stride, uint64_t nD) {
  idtr_local_shape(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}
} // extern "C"

// convert id of our reduction op to id of imex::ptensor reduction op
static ReduceOpId mlir2ddpt(const ::imex::ptensor::ReduceOpId rop) {
  switch (rop) {
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

// convert element type/dtype from MLIR to ddpt
static DTypeId mlir2ddpt(const ::imex::ptensor::DType dt) {
  switch (dt) {
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

/// @return true if size/strides represent a contiguous data layout
bool is_contiguous(const int64_t *sizes, const int64_t *strides, uint64_t nd) {
  if (nd == 0)
    return true;
  if (strides[nd - 1] != 1)
    return false;
  auto sz = 1;
  for (auto i = nd - 1; i > 0; --i) {
    sz *= sizes[i];
    if (strides[i - 1] != sz)
      return false;
  }
  return true;
}

/// copy possibly strided tensor into a contiguous block of data
void bufferize(void *cptr, DTypeId dtype, const int64_t *sizes,
               const int64_t *strides, const int64_t *tStarts,
               const int64_t *tSizes, uint64_t nd, uint64_t N, void *out) {
  dispatch(
      dtype, cptr, [sizes, strides, tStarts, tSizes, nd, N, out](auto *ptr) {
        auto buff = static_cast<decltype(ptr)>(out);

        for (auto i = 0; i < N; ++i) {
          auto szs = &tSizes[i * nd];
          if (szs[0] > 0) {
            auto sts = &tStarts[i * nd];
            uint64_t off = 0;
            for (int64_t r = 0; r < nd; ++r) {
              off += sts[r] * strides[r];
            }
            forall(0, &ptr[off], szs, strides, nd, [&buff](const auto *in) {
              *buff = *in;
              ++buff;
            });
          }
        }
      });
}

template <typename T>
void copy_(uint64_t d, uint64_t &pos, T *cptr, const int64_t *sizes,
           const int64_t *strides, const uint64_t *chunks, uint64_t nd,
           uint64_t start, uint64_t end, T *&out) {
  auto stride = strides[d];
  auto sz = sizes[d];
  auto chunk = chunks[d];
  uint64_t first = 0;
  if (pos < start) {
    first = (start - pos) / chunk;
    pos += first;
    cptr += first * stride;
    assert(pos <= start && pos < end);
  }
  if (d == nd - 1) {
    auto n = std::min(sz - first, end - pos);
    if (stride == 1) {
      memcpy(out, cptr, n * sizeof(T));
    } else {
      for (auto i = 0; i < n; ++i) {
        out[i] = cptr[i * stride];
      }
    }
    pos += n;
    out += n;
  } else {
    for (auto i = first; i < sz; ++i) {
      copy_<T>(d + 1, pos, cptr, sizes, strides, chunks, nd, start, end, out);
      if (pos >= end)
        return;
      cptr += stride;
    }
  }
}

/// copy a number of tensor elements into a contiguous block of data
void bufferizeN(void *cptr, DTypeId dtype, const int64_t *sizes,
                const int64_t *strides, const int64_t *tStarts,
                const int64_t *tEnds, uint64_t nd, uint64_t N, void *out) {
  std::vector<uint64_t> chunks(nd);
  chunks[nd - 1] = 1;
  for (uint64_t i = 1; i < nd; ++i) {
    auto j = nd - i;
    chunks[j - 1] = chunks[j] * sizes[j];
  }
  dispatch(dtype, cptr,
           [sizes, strides, tStarts, tEnds, nd, N, out, &chunks](auto *ptr) {
             auto buff = static_cast<decltype(ptr)>(out);
             for (auto i = 0; i < N; ++i) {
               auto start = tStarts[i];
               auto end = tEnds[i];
               if (end > start) {
                 uint64_t pos = 0;
                 copy_(0, pos, ptr, sizes, strides, chunks.data(), nd, start,
                       end, buff);
               }
             }
           });
}

extern "C" {
// Elementwise inplace allreduce
void idtr_reduce_all(void *inout, DTypeId dtype, uint64_t N, ReduceOpId op) {
  getTransceiver()->reduce_all(inout, dtype, N, op);
}

// FIXME hard-coded for contiguous layout
void _idtr_reduce_all(uint64_t rank, void *data, const int64_t *sizes,
                      const int64_t *strides, int dtype, int op) {
  assert(rank == 0 || strides[rank - 1] == 1);
  idtr_reduce_all(data, mlir2ddpt(static_cast<::imex::ptensor::DType>(dtype)),
                  rank ? rank : 1,
                  mlir2ddpt(static_cast<imex::ptensor::ReduceOpId>(op)));
}

/// @brief reshape tensor
/// We assume tensor is partitioned along the first dimension (only) and
/// partitions are ordered by ranks
/// @param rank
/// @param gShapePtr
/// @param dtype
/// @param lDataPtr
/// @param lOffsPtr
/// @param lShapePtr
/// @param lStridesPtr
/// @param oRank
/// @param oGShapePtr
/// @param oOffsPtr
/// @param oShapePtr
/// @param outPtr
/// @param tc
void _idtr_reshape(int64_t rank, int64_t *gShapePtr, int dtype, void *lDataPtr,
                   int64_t *lOffsPtr, int64_t *lShapePtr, int64_t *lStridesPtr,
                   int64_t oRank, int64_t *oGShapePtr, int64_t *oOffsPtr,
                   int64_t *oShapePtr, void *outPtr, Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif

  assert(std::accumulate(&gShapePtr[0], &gShapePtr[rank], 1,
                         std::multiplies<int64_t>()) ==
         std::accumulate(&oGShapePtr[0], &oGShapePtr[oRank], 1,
                         std::multiplies<int64_t>()));
  assert(std::accumulate(&oOffsPtr[1], &oOffsPtr[oRank], 0,
                         std::plus<int64_t>()) == 0);

  auto N = tc->nranks();
  auto me = tc->rank();
  auto ddpttype = mlir2ddpt(static_cast<::imex::ptensor::DType>(dtype));

  int64_t cSz = std::accumulate(&lShapePtr[1], &lShapePtr[rank], 1,
                                std::multiplies<int64_t>());
  int64_t mySz = cSz * lShapePtr[0];
  int64_t myOff = lOffsPtr[0] * cSz;
  int64_t myEnd = myOff + mySz;
  int64_t tCSz = std::accumulate(&oShapePtr[1], &oShapePtr[oRank], 1,
                                 std::multiplies<int64_t>());
  int64_t myTSz = tCSz * oShapePtr[0];
  int64_t myTOff = oOffsPtr[0] * tCSz;
  int64_t myTEnd = myTOff + myTSz;

  // First we allgather the current and target partitioning

  ::std::vector<int64_t> buff(4 * N);
  buff[me * 4 + 0] = myOff;
  buff[me * 4 + 1] = mySz;
  buff[me * 4 + 2] = myTOff;
  buff[me * 4 + 3] = myTSz;
  ::std::vector<int> counts(N, 4);
  ::std::vector<int> dspl(N);
  for (auto i = 0; i < N; ++i) {
    dspl[i] = 4 * i;
  }
  tc->gather(buff.data(), counts.data(), dspl.data(), ddpttype, REPLICATED);

  // compute overlaps of current parts with requested parts
  // and store meta for alltoall

  std::vector<int> soffs(N, 0);
  std::vector<int> sszs(N, 0);
  std::vector<int> roffs(N, 0);
  std::vector<int> rszs(N, 0);
  std::vector<int64_t> lsOffs(N, 0);
  std::vector<int64_t> lsEnds(N, 0);
  int64_t totSSz = 0;

  for (auto i = 0; i < N; ++i) {
    int64_t *curr = &buff[i * 4];
    auto xOff = curr[0];
    auto xEnd = xOff + curr[1];
    auto tOff = curr[2];
    auto tEnd = tOff + curr[3];

    // first check if this target part overlaps with my local part
    if (tEnd > myOff && tOff < myEnd) {
      auto sOff = std::max(tOff, myOff);
      sszs[i] = std::min(tEnd, myEnd) - sOff;
      soffs[i] = i ? soffs[i - 1] + sszs[i - 1] : 0;
      lsOffs[i] = sOff - myOff;
      lsEnds[i] = lsOffs[i] + sszs[i];
      totSSz += sszs[i];
    }

    // then check if my target part overlaps with the remote local part
    if (myTEnd > xOff && myTOff < xEnd) {
      auto rOff = std::max(xOff, myTOff);
      rszs[i] = std::min(xEnd, myTEnd) - rOff;
      roffs[i] = i ? roffs[i - 1] + rszs[i - 1] : 0;
    }
  }

  Buffer outbuff(totSSz * sizeof_dtype(ddpttype), 2); // FIXME debug value
  bufferizeN(lDataPtr, ddpttype, lShapePtr, lStridesPtr, lsOffs.data(),
             lsEnds.data(), rank, N, outbuff.data());
  tc->alltoall(outbuff.data(), sszs.data(), soffs.data(), ddpttype, outPtr,
               rszs.data(), roffs.data());
}

/// @brief repartition tensor
/// We assume tensor is partitioned along the first dimension (only) and
/// partitions are ordered by ranks
/// @param rank
/// @param gShapePtr
/// @param dtype
/// @param lDataPtr
/// @param lOffsPtr
/// @param lShapePtr
/// @param lStridesPtr
/// @param offsPtr
/// @param szsPtr
/// @param outPtr
/// @param tc
void _idtr_repartition(int64_t rank, int64_t *gShapePtr, int dtype,
                       void *lDataPtr, int64_t *lOffsPtr, int64_t *lShapePtr,
                       int64_t *lStridesPtr, int64_t *offsPtr, int64_t *szsPtr,
                       void *outPtr, Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif
  auto N = tc->nranks();
  auto me = tc->rank();
  auto ddpttype = mlir2ddpt(static_cast<::imex::ptensor::DType>(dtype));

  // First we allgather the requested target partitioning

  auto myBOff = 2 * rank * me;
  ::std::vector<int64_t> buff(2 * rank * N);
  for (int64_t i = 0; i < rank; ++i) {
    // assert(offsPtr[i] - lOffsPtr[i] + szsPtr[i] <= gShapePtr[i]);
    buff[myBOff + i] = offsPtr[i];
    buff[myBOff + i + rank] = szsPtr[i];
  }
  ::std::vector<int> counts(N, rank * 2);
  ::std::vector<int> dspl(N);
  for (auto i = 0; i < N; ++i) {
    dspl[i] = 2 * rank * i;
  }
  tc->gather(buff.data(), counts.data(), dspl.data(), ddpttype, REPLICATED);

  // compute overlap of my local data with each requested part

  auto myOff = lOffsPtr[0];
  auto mySz = lShapePtr[0];
  auto myEnd = myOff + mySz;
  auto myTileSz = std::accumulate(&lShapePtr[1], &lShapePtr[rank], 1,
                                  std::multiplies<int64_t>());

  std::vector<int> soffs(N);
  std::vector<int> sszs(N, 0);
  std::vector<int64_t> tStarts(N * rank, 0);
  std::vector<int64_t> tSizes(N * rank, 0);
  std::vector<int64_t> nSizes(N);
  int64_t totSSz = 0;
  bool needsBufferize = !is_contiguous(lShapePtr, lStridesPtr, rank);

  for (auto i = 0; i < N; ++i) {
    nSizes[i] = std::accumulate(&buff[2 * rank * i + rank + 1],
                                &buff[2 * rank * i + rank + rank], 1,
                                std::multiplies<int64_t>());
    if (nSizes[i] != myTileSz)
      needsBufferize = true;
  }
  for (auto i = 0; i < N; ++i) {
    auto nSz = nSizes[i];
    auto tOff = buff[2 * rank * i];
    auto tSz = buff[2 * rank * i + rank];
    auto tEnd = tOff + tSz;

    if (tEnd > myOff && tOff < myEnd) {
      // We have a target partition which is inside my local data
      // we now compute what data goes to this target partition
      auto start = std::max(myOff, tOff);
      auto end = std::min(myEnd, tEnd);
      tStarts[i * rank] = start - myOff;
      tSizes[i * rank] = end - start;
      soffs[i] = needsBufferize ? (i ? soffs[i - 1] + sszs[i - 1] : 0)
                                : (int)(start - myOff) * myTileSz;
      sszs[i] = (int)(end - start) * nSz;
    } else {
      soffs[i] = i ? soffs[i - 1] + sszs[i - 1] : 0;
    }
    totSSz += sszs[i];
    for (auto r = 1; r < rank; ++r) {
      tStarts[i * rank + r] = buff[2 * rank * i + r];
      tSizes[i * rank + r] = buff[2 * rank * i + rank + r];
      // assert(tSizes[i*rank+r] <= lShapePtr[r]);
    }
  }

  // send our send sizes to others and receive theirs
  std::vector<int> rszs(N);
  tc->alltoall(sszs.data(), 1, INT32, rszs.data());

  // compute receive-displacements
  std::vector<int> roffs(N);
  roffs[0] = 0;
  for (auto i = 1; i < N; ++i) {
    // compute for all i > 0
    roffs[i] = roffs[i - 1] + rszs[i - 1];
  }

  // Finally communicate elements
  if (needsBufferize) {
    // create send buffer if strided
    Buffer buff(totSSz * sizeof_dtype(ddpttype), 2);
    bufferize(lDataPtr, ddpttype, lShapePtr, lStridesPtr, tStarts.data(),
              tSizes.data(), rank, N, buff.data());
    tc->alltoall(buff.data(), sszs.data(), soffs.data(), ddpttype, outPtr,
                 rszs.data(), roffs.data());
  } else {
    tc->alltoall(lDataPtr, sszs.data(), soffs.data(), ddpttype, outPtr,
                 rszs.data(), roffs.data());
  }
}

// debug helper
void _idtr_extractslice(int64_t *slcOffs, int64_t *slcSizes,
                        int64_t *slcStrides, int64_t *tOffs, int64_t *tSizes,
                        int64_t *lSlcOffsets, int64_t *lSlcSizes,
                        int64_t *gSlcOffsets) {
  if (slcOffs)
    std::cerr << "slcOffs: " << slcOffs[0] << " " << slcOffs[1] << std::endl;
  if (slcSizes)
    std::cerr << "slcSizes: " << slcSizes[0] << " " << slcSizes[1] << std::endl;
  if (slcStrides)
    std::cerr << "slcStrides: " << slcStrides[0] << " " << slcStrides[1]
              << std::endl;
  if (tOffs)
    std::cerr << "tOffs: " << tOffs[0] << " " << tOffs[1] << std::endl;
  if (tSizes)
    std::cerr << "tSizes: " << tSizes[0] << " " << tSizes[1] << std::endl;
  if (lSlcOffsets)
    std::cerr << "lSlcOffsets: " << lSlcOffsets[0] << " " << lSlcOffsets[1]
              << std::endl;
  if (lSlcSizes)
    std::cerr << "lSlcSizes: " << lSlcSizes[0] << " " << lSlcSizes[1]
              << std::endl;
  if (gSlcOffsets)
    std::cerr << "gSlcOffsets: " << gSlcOffsets[0] << " " << gSlcOffsets[1]
              << std::endl;
}

void _debugFunc() { std::cerr << "_debugfunc\n"; }
} // extern "C"
