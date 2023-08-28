// SPDX-License-Identifier: BSD-3-Clause

/*
    Intel Distributed Runtime for MLIR
*/

#include <ddptensor/DDPTensorImpl.hpp>
#include <ddptensor/MPITransceiver.hpp>
#include <ddptensor/MemRefType.hpp>

#include <imex/Dialect/PTensor/IR/PTensorDefs.h>

#include <cassert>
#include <iostream>
#include <memory>

constexpr id_t UNKNOWN_GUID = -1;

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
    pos += first * chunk;
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

using MRIdx1d = Unranked1DMemRefType<int64_t>;

// FIXME hard-coded for contiguous layout
template <typename T>
void _idtr_reduce_all(int64_t dataRank, void *dataDescr, int op) {
  UnrankedMemRefType<T> data(dataRank, dataDescr);
  assert(dataRank == 0 || (dataRank == 1 && data.strides()[0] == 1));
  auto d = data.data();
  auto t = DTYPE<T>::value;
  auto r = dataRank ? data.sizes()[0] : 1;
  auto o = mlir2ddpt(static_cast<imex::ptensor::ReduceOpId>(op));
  getTransceiver()->reduce_all(d, t, r, o);
}

extern "C" {

#define TYPED_REDUCEALL(_sfx, _typ)                                            \
  void _idtr_reduce_all_##_sfx(int64_t dataRank, void *dataDescr, int op) {    \
    _idtr_reduce_all<_typ>(dataRank, dataDescr, op);                           \
  }

TYPED_REDUCEALL(f64, double);
TYPED_REDUCEALL(f32, float);
TYPED_REDUCEALL(i64, int64_t);
TYPED_REDUCEALL(i32, int32_t);
TYPED_REDUCEALL(i16, int16_t);
TYPED_REDUCEALL(i8, int8_t);
TYPED_REDUCEALL(i1, bool);

} // extern "C"

/// @brief reshape tensor
/// We assume tensor is partitioned along the first dimension (only) and
/// partitions are ordered by ranks
void _idtr_reshape(DTypeId ddpttype, int64_t lRank, int64_t *gShapePtr,
                   void *lDataPtr, int64_t *lShapePtr, int64_t *lStridesPtr,
                   int64_t *lOffsPtr, int64_t oRank, int64_t *oGShapePtr,
                   void *oDataPtr, int64_t *oShapePtr, int64_t *oOffsPtr,
                   Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif

  assert(std::accumulate(&gShapePtr[0], &gShapePtr[lRank], 1,
                         std::multiplies<int64_t>()) ==
         std::accumulate(&oGShapePtr[0], &oGShapePtr[oRank], 1,
                         std::multiplies<int64_t>()));
  assert(std::accumulate(&oOffsPtr[1], &oOffsPtr[oRank], 0,
                         std::plus<int64_t>()) == 0);

  auto N = tc->nranks();
  auto me = tc->rank();

  int64_t cSz = std::accumulate(&lShapePtr[1], &lShapePtr[lRank], 1,
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
  tc->gather(buff.data(), counts.data(), dspl.data(), INT64, REPLICATED);

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
             lsEnds.data(), lRank, N, outbuff.data());
  tc->alltoall(outbuff.data(), sszs.data(), soffs.data(), ddpttype, oDataPtr,
               rszs.data(), roffs.data());
}

/// @brief reshape tensor
template <typename T>
void _idtr_reshape(int64_t gShapeRank, void *gShapeDescr, int64_t lOffsRank,
                   void *lOffsDescr, int64_t lRank, void *lDescr,
                   int64_t oGShapeRank, void *oGShapeDescr, int64_t oOffsRank,
                   void *oOffsDescr, int64_t oRank, void *oDescr,
                   Transceiver *tc) {

  auto ddpttype = DTYPE<T>::value;

  UnrankedMemRefType<T> lData(lRank, lDescr);
  UnrankedMemRefType<T> oData(oRank, oDescr);

  _idtr_reshape(ddpttype, lRank, MRIdx1d(gShapeRank, gShapeDescr).data(),
                lData.data(), lData.sizes(), lData.strides(),
                MRIdx1d(lOffsRank, lOffsDescr).data(), oRank,
                MRIdx1d(oGShapeRank, oGShapeDescr).data(), oData.data(),
                oData.sizes(), MRIdx1d(oOffsRank, oOffsDescr).data(), tc);
}

extern "C" {

#define TYPED_RESHAPE(_sfx, _typ)                                              \
  void _idtr_reshape_##_sfx(                                                   \
      int64_t gShapeRank, void *gShapeDescr, int64_t lOffsRank,                \
      void *lOffsDescr, int64_t rank, void *lDescr, int64_t oGShapeRank,       \
      void *oGShapeDescr, int64_t oOffsRank, void *oOffsDescr, int64_t oRank,  \
      void *oDescr, Transceiver *tc) {                                         \
    _idtr_reshape<_typ>(gShapeRank, gShapeDescr, lOffsRank, lOffsDescr, rank,  \
                        lDescr, oGShapeRank, oGShapeDescr, oOffsRank,          \
                        oOffsDescr, oRank, oDescr, tc);                        \
  }

TYPED_RESHAPE(f64, double);
TYPED_RESHAPE(f32, float);
TYPED_RESHAPE(i64, int64_t);
TYPED_RESHAPE(i32, int32_t);
TYPED_RESHAPE(i16, int16_t);
TYPED_RESHAPE(i8, int8_t);
TYPED_RESHAPE(i1, bool);

} // extern "C"

/// @brief Update data in halo parts
/// We assume tensor is partitioned along the first dimension only
/// (row partitioning) and partitions are ordered by ranks
void _idtr_update_halo(DTypeId ddpttype, int64_t ndims, int64_t *ownedOff,
                       int64_t *ownedShape, int64_t *ownedStride,
                       int64_t *bbOff, int64_t *bbShape, void *ownedData,
                       void *leftHaloData, void *rightHaloData,
                       Transceiver *tc) {

#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = getTransceiver();
#endif
  auto nworkers = tc->nranks();
  if (nworkers <= 1)
    return;
  auto myWorkerIndex = tc->rank();

  // Gather table with bounding box offsets and shapes for all workers
  // [ (w0 offsets) o_0, o_1, ..., o_ndims,
  //   (w0  shapes) s_0, s_1, ..., s_ndims,
  //   (w1 offsets) ... ]
  ::std::vector<int64_t> bbTable(2 * ndims * nworkers);
  auto ptableStart = 2 * ndims * myWorkerIndex;
  for (int64_t i = 0; i < ndims; ++i) {
    bbTable[ptableStart + i] = bbOff[i];
    bbTable[ptableStart + i + ndims] = bbShape[i];
  }
  ::std::vector<int> counts(nworkers, ndims * 2);
  ::std::vector<int> offsets(nworkers);
  for (auto i = 0; i < nworkers; ++i) {
    offsets[i] = 2 * ndims * i;
  }
  tc->gather(bbTable.data(), counts.data(), offsets.data(), INT64, REPLICATED);

  // global indices for row partitioning
  auto ownedRowStart = ownedOff[0];
  auto ownedRows = ownedShape[0];
  auto ownedRowEnd = ownedRowStart + ownedRows;
  // all remaining dims are treated as one large column
  auto ownedTotCols = std::accumulate(&ownedShape[1], &ownedShape[ndims], 1,
                                      std::multiplies<int64_t>());
  auto bbTotCols = std::accumulate(&bbShape[1], &bbShape[ndims], 1,
                                   std::multiplies<int64_t>());

  // find local elements to send to next workers (destination leftHalo)
  // and previous workers (destination rightHalo)
  std::vector<int> lSendOff(nworkers, 0), rSendOff(nworkers, 0);
  std::vector<int> lSendSize(nworkers, 0), rSendSize(nworkers, 0);

  // use send buffer if owned data is strided or sending a subview
  bool bufferizeSend = (!is_contiguous(ownedShape, ownedStride, ndims) ||
                        bbTotCols != ownedTotCols);

  // assert(!bufferizeSend);
  std::vector<int64_t> lBufferStart(nworkers * ndims, 0);
  std::vector<int64_t> lBufferSize(nworkers * ndims, 0);
  std::vector<int64_t> rBufferStart(nworkers * ndims, 0);
  std::vector<int64_t> rBufferSize(nworkers * ndims, 0);
  int64_t lTotalSendSize = 0, rTotalSendSize = 0;

  for (auto i = 0; i < nworkers; ++i) {
    if (i == myWorkerIndex) {
      continue;
    }
    // worker i bounding box indices
    auto bRowStart = bbTable[2 * ndims * i];
    auto bRows = bbTable[2 * ndims * i + ndims];
    auto bRowEnd = bRowStart + bRows;

    if (bRowEnd > ownedRowStart && bRowStart < ownedRowEnd) {
      // bounding box overlaps with local data
      // calculate indices for data to be sent
      auto globalRowStart = std::max(ownedRowStart, bRowStart);
      auto globalRowEnd = std::min(ownedRowEnd, bRowEnd);
      auto localRowStart = globalRowStart - ownedRowStart;
      auto localStart = (int)(localRowStart)*ownedTotCols;
      auto nRows = globalRowEnd - globalRowStart;
      auto nSend = (int)(nRows)*bbTotCols;

      if (i < myWorkerIndex) {
        // target is rightHalo
        if (bufferizeSend) {
          rSendOff[i] = i ? rSendOff[i - 1] + rSendSize[i - 1] : 0;
          rBufferStart[i * ndims] = localRowStart;
          rBufferSize[i * ndims] = nRows;
          for (auto j = 1; j < ndims; ++j) {
            rBufferStart[i * ndims + j] = bbOff[j];
            rBufferSize[i * ndims + j] = bbShape[j];
          }
        } else {
          rSendOff[i] = localStart;
        }
        rSendSize[i] = nSend;
        rTotalSendSize += nSend;
      } else {
        // target is leftHalo
        if (bufferizeSend) {
          lSendOff[i] = i ? lSendOff[i - 1] + lSendSize[i - 1] : 0;
          lBufferStart[i * ndims] = localRowStart;
          lBufferSize[i * ndims] = nRows;
          for (auto j = 1; j < ndims; ++j) {
            lBufferStart[i * ndims + j] = bbOff[j];
            lBufferSize[i * ndims + j] = bbShape[j];
          }
        } else {
          lSendOff[i] = localStart;
        }
        lSendSize[i] = nSend;
        lTotalSendSize += nSend;
      }
    }
  }

  // receive maps
  std::vector<int> lRecvSize(nworkers), rRecvSize(nworkers);
  std::vector<int> lRecvOff(nworkers), rRecvOff(nworkers);
  // receive size is sender's send size
  tc->alltoall(lSendSize.data(), 1, INT32, lRecvSize.data());
  tc->alltoall(rSendSize.data(), 1, INT32, rRecvSize.data());
  // compute offset in a contiguous receive buffer
  lRecvOff[0] = 0;
  rRecvOff[0] = 0;
  for (auto i = 1; i < nworkers; ++i) {
    lRecvOff[i] = lRecvOff[i - 1] + lRecvSize[i - 1];
    rRecvOff[i] = rRecvOff[i - 1] + rRecvSize[i - 1];
  }

  // communicate left/right halos
  if (bufferizeSend) {
    Buffer sendBuff;
    sendBuff.resize(std::max(lTotalSendSize, rTotalSendSize) *
                    sizeof_dtype(ddpttype));
    bufferize(ownedData, ddpttype, ownedShape, ownedStride, lBufferStart.data(),
              lBufferSize.data(), ndims, nworkers, sendBuff.data());
    tc->alltoall(sendBuff.data(), lSendSize.data(), lSendOff.data(), ddpttype,
                 leftHaloData, lRecvSize.data(), lRecvOff.data());
    bufferize(ownedData, ddpttype, ownedShape, ownedStride, rBufferStart.data(),
              rBufferSize.data(), ndims, nworkers, sendBuff.data());
    tc->alltoall(sendBuff.data(), rSendSize.data(), rSendOff.data(), ddpttype,
                 rightHaloData, rRecvSize.data(), rRecvOff.data());
  } else {
    tc->alltoall(ownedData, lSendSize.data(), lSendOff.data(), ddpttype,
                 leftHaloData, lRecvSize.data(), lRecvOff.data());
    tc->alltoall(ownedData, rSendSize.data(), rSendOff.data(), ddpttype,
                 rightHaloData, rRecvSize.data(), rRecvOff.data());
  }
}

/// @brief templated wrapper for typed function versions calling
/// _idtr_update_halo
template <typename T>
void _idtr_update_halo(Transceiver *tc, int64_t gShapeRank, void *gShapeDescr,
                       int64_t oOffRank, void *oOffDescr, int64_t oDataRank,
                       void *oDataDescr, int64_t bbOffRank, void *bbOffDescr,
                       int64_t bbShapeRank, void *bbShapeDescr,
                       int64_t lHaloRank, void *lHaloDescr, int64_t rHaloRank,
                       void *rHaloDescr) {

  auto ddpttype = DTYPE<T>::value;

  // Construct unranked memrefs for metadata and data
  MRIdx1d ownedOff(oOffRank, oOffDescr);
  MRIdx1d bbOff(bbOffRank, bbOffDescr);
  MRIdx1d bbShape(bbShapeRank, bbShapeDescr);
  UnrankedMemRefType<T> ownedData(oDataRank, oDataDescr);
  UnrankedMemRefType<T> leftHalo(lHaloRank, lHaloDescr);
  UnrankedMemRefType<T> rightHalo(rHaloRank, rHaloDescr);

  assert(leftHalo.contiguous_layout());
  assert(rightHalo.contiguous_layout());

  _idtr_update_halo(ddpttype, ownedData.rank(), ownedOff.data(),
                    ownedData.sizes(), ownedData.strides(), bbOff.data(),
                    bbShape.data(), ownedData.data(), leftHalo.data(),
                    rightHalo.data(), tc);
}

extern "C" {
#define TYPED_UPDATE_HALO(_sfx, _typ)                                          \
  void _idtr_update_halo_##_sfx(                                               \
      Transceiver *tc, int64_t gShapeRank, void *gShapeDescr,                  \
      int64_t oOffRank, void *oOffDescr, int64_t oDataRank, void *oDataDescr,  \
      int64_t bbOffRank, void *bbOffDescr, int64_t bbShapeRank,                \
      void *bbShapeDescr, int64_t lHaloRank, void *lHaloDescr,                 \
      int64_t rHaloRank, void *rHaloDescr) {                                   \
    _idtr_update_halo<_typ>(tc, gShapeRank, gShapeDescr, oOffRank, oOffDescr,  \
                            oDataRank, oDataDescr, bbOffRank, bbOffDescr,      \
                            bbShapeRank, bbShapeDescr, lHaloRank, lHaloDescr,  \
                            rHaloRank, rHaloDescr);                            \
  }

TYPED_UPDATE_HALO(f64, double);
TYPED_UPDATE_HALO(f32, float);
TYPED_UPDATE_HALO(i64, int64_t);
TYPED_UPDATE_HALO(i32, int32_t);
TYPED_UPDATE_HALO(i16, int16_t);
TYPED_UPDATE_HALO(i8, int8_t);
TYPED_UPDATE_HALO(i1, bool);

} // extern "C"

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

extern "C" {
void _debugFunc() { std::cerr << "_debugfunc\n"; }
} // extern "C"
