// SPDX-License-Identifier: BSD-3-Clause

/*
    Intel Distributed Runtime for MLIR
*/

#include <sharpy/MPITransceiver.hpp>
#include <sharpy/MemRefType.hpp>
#include <sharpy/NDArray.hpp>

#include <imex/Dialect/NDArray/IR/NDArrayDefs.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_map>

#define STRINGIFY(a) #a

constexpr id_t UNKNOWN_GUID = -1;

using container_type =
    std::unordered_map<SHARPY::id_type, std::unique_ptr<SHARPY::NDArray>>;

static container_type garrays;
static SHARPY::id_type _nguid = -1;
inline SHARPY::id_type get_guid() { return ++_nguid; }

// Transceiver * theTransceiver = MPITransceiver();

template <typename T> T *mr_to_ptr(void *ptr, intptr_t offset) {
  if (!ptr) {
    throw std::runtime_error("Fatal: cannot handle offset on nullptr");
  }
  return reinterpret_cast<T *>(ptr) + offset;
}

// abstract handle providing an abstract wait method
struct WaitHandleBase {
  virtual ~WaitHandleBase(){};
  virtual void wait() = 0;
};

// concrete handle to be instantiated with a lambda or alike
// the lambda will be executed within wait()
template <typename T> class WaitHandle : public WaitHandleBase {
  T _fini;

public:
  WaitHandle(T fini) : _fini(fini) {}
  virtual void wait() override { _fini(); }
};

template <typename T> WaitHandle<T> *mkWaitHandle(T fini) {
  return new WaitHandle<T>(fini);
};

void _idtr_wait(WaitHandleBase *handle, int64_t lHaloRank, void *lHaloDescr,
                int64_t rHaloRank, void *rHaloDescr) {
  if (handle) {
    handle->wait();
    delete handle;
  }
}

extern "C" {
#define TYPED_WAIT(_sfx)                                                       \
  void _idtr_wait_##_sfx(WaitHandleBase *handle, int64_t lHaloRank,            \
                         void *lHaloDescr, int64_t rHaloRank,                  \
                         void *rHaloDescr) {                                   \
    return _idtr_wait(handle, lHaloRank, lHaloDescr, rHaloRank, rHaloDescr);   \
  }                                                                            \
  _Pragma(STRINGIFY(weak _mlir_ciface__idtr_wait_##_sfx = _idtr_wait_##_sfx))

TYPED_WAIT(f64);
TYPED_WAIT(f32);
TYPED_WAIT(i64);
TYPED_WAIT(i32);
TYPED_WAIT(i16);
TYPED_WAIT(i8);
TYPED_WAIT(i1);

#define NO_TRANSCEIVER
#ifdef NO_TRANSCEIVER
static void initMPIRuntime() {
  if (SHARPY::getTransceiver() == nullptr)
    SHARPY::init_transceiver(new SHARPY::MPITransceiver(false));
}
#endif

// Return number of ranks/processes in given team/communicator
uint64_t idtr_nprocs(SHARPY::Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = SHARPY::getTransceiver();
#endif
  return tc ? tc->nranks() : 1;
}
#pragma weak _idtr_nprocs = idtr_nprocs
#pragma weak _mlir_ciface__idtr_nprocs = idtr_nprocs

// Return rank in given team/communicator
uint64_t idtr_prank(SHARPY::Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = SHARPY::getTransceiver();
#endif
  return tc ? tc->rank() : 0;
}
#pragma weak _idtr_prank = idtr_prank
#pragma weak _mlir_ciface__idtr_prank = idtr_prank

// Register a global array of given shape.
// Returns guid.
// The runtime does not own or manage any memory.
id_t idtr_init_array(const uint64_t *shape, uint64_t nD) {
  auto guid = get_guid();
  // garrays[guid] = std::unique_ptr<SHARPY::NDArray>(nD ? new
  // SHARPY::NDArray(shape, nD) : new SHARPY::NDArray);
  return guid;
}

id_t _idtr_init_array(void *alloced, void *aligned, intptr_t offset,
                      intptr_t size, intptr_t stride, uint64_t nD) {
  return idtr_init_array(mr_to_ptr<uint64_t>(aligned, offset), nD);
}

// Get the offsets (one for each dimension) of the local partition of a
// distributed array in number of elements. Result is stored in provided array.
void idtr_local_offsets(id_t guid, uint64_t *offsets, uint64_t nD) {
#if 0
    const auto & tnsr = garrays.at(guid);
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
// distributed array in number of elements. Result is stored in provided array.
void idtr_local_shape(id_t guid, uint64_t *lshape, uint64_t N) {
#if 0
    const auto & tnsr = garrays.at(guid);
    auto shp = tnsr->slice().local_slice().shape();
    std::copy(shp.begin(), shp.end(), lshape);
#endif
}

void _idtr_local_shape(id_t guid, void *alloced, void *aligned, intptr_t offset,
                       intptr_t size, intptr_t stride, uint64_t nD) {
  idtr_local_shape(guid, mr_to_ptr<uint64_t>(aligned, offset), nD);
}
} // extern "C"

// convert id of our reduction op to id of imex::ndarray reduction op
static SHARPY::ReduceOpId mlir2sharpy(const ::imex::ndarray::ReduceOpId rop) {
  switch (rop) {
  case ::imex::ndarray::MEAN:
    return SHARPY::MEAN;
  case ::imex::ndarray::PROD:
    return SHARPY::PROD;
  case ::imex::ndarray::SUM:
    return SHARPY::SUM;
  case ::imex::ndarray::STD:
    return SHARPY::STD;
  case ::imex::ndarray::VAR:
    return SHARPY::VAR;
  case ::imex::ndarray::MAX:
    return SHARPY::MAX;
  case ::imex::ndarray::MIN:
    return SHARPY::MIN;
  default:
    throw std::runtime_error("Unknown reduction operation");
  }
}

// convert element type/dtype from MLIR to sharpy
[[maybe_unused]] static SHARPY::DTypeId
mlir2sharpy(const ::imex::ndarray::DType dt) {
  switch (dt) {
  case ::imex::ndarray::DType::F64:
    return SHARPY::FLOAT64;
    break;
  case ::imex::ndarray::DType::I64:
    return SHARPY::INT64;
    break;
  case ::imex::ndarray::DType::U64:
    return SHARPY::UINT64;
    break;
  case ::imex::ndarray::DType::F32:
    return SHARPY::FLOAT32;
    break;
  case ::imex::ndarray::DType::I32:
    return SHARPY::INT32;
    break;
  case ::imex::ndarray::DType::U32:
    return SHARPY::UINT32;
    break;
  case ::imex::ndarray::DType::I16:
    return SHARPY::INT16;
    break;
  case ::imex::ndarray::DType::U16:
    return SHARPY::UINT16;
    break;
  case ::imex::ndarray::DType::I8:
    return SHARPY::INT8;
    break;
  case ::imex::ndarray::DType::U8:
    return SHARPY::UINT8;
    break;
  case ::imex::ndarray::DType::I1:
    return SHARPY::BOOL;
    break;
  default:
    throw std::runtime_error("unknown dtype");
  };
}

/// copy possibly strided array into a contiguous block of data
void bufferize(void *cptr, SHARPY::DTypeId dtype, const int64_t *sizes,
               const int64_t *strides, const int64_t *tStarts,
               const int64_t *tSizes, uint64_t nd, uint64_t N, void *out) {
  if (!cptr || !sizes || !strides || !tStarts || !tSizes) {
    return;
  }
  dispatch(dtype, cptr,
           [sizes, strides, tStarts, tSizes, nd, N, out](auto *ptr) {
             auto buff = static_cast<decltype(ptr)>(out);

             for (auto i = 0ul; i < N; ++i) {
               auto szs = &tSizes[i * nd];
               if (szs[0] > 0) {
                 auto sts = &tStarts[i * nd];
                 uint64_t off = 0;
                 for (auto r = 0ul; r < nd; ++r) {
                   off += sts[r] * strides[r];
                 }
                 SHARPY::forall(0, &ptr[off], szs, strides, nd,
                                [&buff](const auto *in) {
                                  *buff = *in;
                                  ++buff;
                                });
               }
             }
           });
}

/// copy contiguous block of data into a possibly strided array
void unpack(void *in, SHARPY::DTypeId dtype, const int64_t *sizes,
            const int64_t *strides, const int64_t *tStarts,
            const int64_t *tSizes, uint64_t nd, uint64_t N, void *out) {
  if (!in || !sizes || !strides || !tStarts || !tSizes || !out) {
    return;
  }
  dispatch(dtype, out, [sizes, strides, tStarts, tSizes, nd, N, in](auto *ptr) {
    auto buff = static_cast<decltype(ptr)>(in);

    for (auto i = 0ul; i < N; ++i) {
      auto szs = &tSizes[i * nd];
      if (szs[0] > 0) {
        auto sts = &tStarts[i * nd];
        uint64_t off = 0;
        for (auto r = 0ul; r < nd; ++r) {
          off += sts[r] * strides[r];
        }
        SHARPY::forall(0, &ptr[off], szs, strides, nd, [&buff](auto *out) {
          *out = *buff;
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
  if (!cptr || !sizes || !strides || !chunks || !out) {
    return;
  }
  auto stride = strides[d];
  uint64_t sz = sizes[d];
  uint64_t chunk = chunks[d];
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
      for (auto i = 0ul; i < n; ++i) {
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

/// copy a number of array elements into a contiguous block of data
void bufferizeN(void *cptr, SHARPY::DTypeId dtype, const int64_t *sizes,
                const int64_t *strides, const int64_t *tStarts,
                const int64_t *tEnds, uint64_t nd, uint64_t N, void *out) {
  if (!cptr || !sizes || !strides || !tStarts || !tEnds || !out) {
    return;
  }
  std::vector<uint64_t> chunks(nd);
  chunks[nd - 1] = 1;
  for (uint64_t i = 1; i < nd; ++i) {
    auto j = nd - i;
    chunks[j - 1] = chunks[j] * sizes[j];
  }
  dispatch(dtype, cptr,
           [sizes, strides, tStarts, tEnds, nd, N, out, &chunks](auto *ptr) {
             auto buff = static_cast<decltype(ptr)>(out);
             for (auto i = 0ul; i < N; ++i) {
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

using MRIdx1d = SHARPY::Unranked1DMemRefType<int64_t>;

// FIXME hard-coded for contiguous layout
template <typename T>
void _idtr_reduce_all(int64_t dataRank, void *dataDescr, int op) {
  auto tc = SHARPY::getTransceiver();
  if (!tc)
    return;
  SHARPY::UnrankedMemRefType<T> data(dataRank, dataDescr);
  assert(dataRank == 0 || (dataRank == 1 && data.strides()[0] == 1));
  auto d = data.data();
  auto t = SHARPY::DTYPE<T>::value;
  auto r = dataRank ? data.sizes()[0] : 1;
  auto o = mlir2sharpy(static_cast<imex::ndarray::ReduceOpId>(op));
  tc->reduce_all(d, t, r, o);
}

extern "C" {

#define TYPED_REDUCEALL(_sfx, _typ)                                            \
  void _idtr_reduce_all_##_sfx(int64_t dataRank, void *dataDescr, int op) {    \
    _idtr_reduce_all<_typ>(dataRank, dataDescr, op);                           \
  }                                                                            \
  _Pragma(STRINGIFY(weak _mlir_ciface__idtr_reduce_all_##_sfx =                \
                        _idtr_reduce_all_##_sfx))

TYPED_REDUCEALL(f64, double);
TYPED_REDUCEALL(f32, float);
TYPED_REDUCEALL(i64, int64_t);
TYPED_REDUCEALL(i32, int32_t);
TYPED_REDUCEALL(i16, int16_t);
TYPED_REDUCEALL(i8, int8_t);
TYPED_REDUCEALL(i1, bool);

} // extern "C"

/// @brief reshape array
/// We assume array is partitioned along the first dimension (only) and
/// partitions are ordered by ranks
void _idtr_reshape(SHARPY::DTypeId sharpytype, int64_t lRank,
                   int64_t *gShapePtr, void *lDataPtr, int64_t *lShapePtr,
                   int64_t *lStridesPtr, int64_t *lOffsPtr, int64_t oRank,
                   int64_t *oGShapePtr, void *oDataPtr, int64_t *oShapePtr,
                   int64_t *oOffsPtr, SHARPY::Transceiver *tc) {
#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = SHARPY::getTransceiver();
#endif
  if (!gShapePtr || !lDataPtr || !lShapePtr || !lStridesPtr || !lOffsPtr ||
      !oGShapePtr || !oDataPtr || !oShapePtr || !oOffsPtr || !tc) {
    throw std::runtime_error("Fatal: received nullptr in reshape");
  }

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
  for (auto i = 0ul; i < N; ++i) {
    dspl[i] = 4 * i;
  }
  tc->gather(buff.data(), counts.data(), dspl.data(), SHARPY::INT64,
             SHARPY::REPLICATED);

  // compute overlaps of current parts with requested parts
  // and store meta for alltoall

  std::vector<int> soffs(N, 0);
  std::vector<int> sszs(N, 0);
  std::vector<int> roffs(N, 0);
  std::vector<int> rszs(N, 0);
  std::vector<int64_t> lsOffs(N, 0);
  std::vector<int64_t> lsEnds(N, 0);
  int64_t totSSz = 0;

  for (auto i = 0ul; i < N; ++i) {
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

  SHARPY::Buffer outbuff(totSSz * sizeof_dtype(sharpytype),
                         2); // FIXME debug value
  bufferizeN(lDataPtr, sharpytype, lShapePtr, lStridesPtr, lsOffs.data(),
             lsEnds.data(), lRank, N, outbuff.data());
  auto hdl = tc->alltoall(outbuff.data(), sszs.data(), soffs.data(), sharpytype,
                          oDataPtr, rszs.data(), roffs.data());
  tc->wait(hdl);
}

/// @brief reshape array
template <typename T>
void _idtr_reshape(int64_t gShapeRank, void *gShapeDescr, int64_t lOffsRank,
                   void *lOffsDescr, int64_t lRank, void *lDescr,
                   int64_t oGShapeRank, void *oGShapeDescr, int64_t oOffsRank,
                   void *oOffsDescr, int64_t oRank, void *oDescr,
                   SHARPY::Transceiver *tc) {

  auto sharpytype = SHARPY::DTYPE<T>::value;

  SHARPY::UnrankedMemRefType<T> lData(lRank, lDescr);
  SHARPY::UnrankedMemRefType<T> oData(oRank, oDescr);

  _idtr_reshape(sharpytype, lRank, MRIdx1d(gShapeRank, gShapeDescr).data(),
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
      void *oDescr, SHARPY::Transceiver *tc) {                                 \
    _idtr_reshape<_typ>(gShapeRank, gShapeDescr, lOffsRank, lOffsDescr, rank,  \
                        lDescr, oGShapeRank, oGShapeDescr, oOffsRank,          \
                        oOffsDescr, oRank, oDescr, tc);                        \
  }                                                                            \
  _Pragma(STRINGIFY(weak _mlir_ciface__idtr_reshape_##_sfx =                   \
                        _idtr_reshape_##_sfx))

TYPED_RESHAPE(f64, double);
TYPED_RESHAPE(f32, float);
TYPED_RESHAPE(i64, int64_t);
TYPED_RESHAPE(i32, int32_t);
TYPED_RESHAPE(i16, int16_t);
TYPED_RESHAPE(i8, int8_t);
TYPED_RESHAPE(i1, bool);

} // extern "C"

// struct for caching meta data for update_halo
// no copies allowed, only move-semantics and reference access
struct UHCache {
  // copying needed?
  std::vector<int64_t> _lBufferStart, _lBufferSize, _rBufferStart, _rBufferSize;
  std::vector<int64_t> _lRecvBufferSize, _rRecvBufferSize;
  // send maps
  std::vector<int> _lSendSize, _rSendSize, _lSendOff, _rSendOff;
  // receive maps
  std::vector<int> _lRecvSize, _rRecvSize, _lRecvOff, _rRecvOff;
  // buffers
  SHARPY::Buffer _recvLBuff, _recvRBuff, _sendLBuff, _sendRBuff;
  bool _bufferizeSend, _bufferizeLRecv, _bufferizeRRecv;
  // start and sizes for chunks from remotes if copies are needed
  int64_t _lTotalRecvSize, _rTotalRecvSize, _lTotalSendSize, _rTotalSendSize;

  UHCache() = default;
  UHCache(const UHCache &) = delete;
  UHCache(UHCache &&) = default;
  UHCache(std::vector<int64_t> &&lBufferStart,
          std::vector<int64_t> &&lBufferSize,
          std::vector<int64_t> &&rBufferStart,
          std::vector<int64_t> &&rBufferSize,
          std::vector<int64_t> &&lRecvBufferSize,
          std::vector<int64_t> &&rRecvBufferSize, std::vector<int> &&lSendSize,
          std::vector<int> &&rSendSize, std::vector<int> &&lSendOff,
          std::vector<int> &&rSendOff, std::vector<int> &&lRecvSize,
          std::vector<int> &&rRecvSize, std::vector<int> &&lRecvOff,
          SHARPY::Buffer &&recvLBuff, SHARPY::Buffer &&recvRBuff,
          SHARPY::Buffer &&sendLBuff, SHARPY::Buffer &&sendRBuff,
          std::vector<int> &&rRecvOff, bool bufferizeSend, bool bufferizeLRecv,
          bool bufferizeRRecv, int64_t lTotalRecvSize, int64_t rTotalRecvSize,
          int64_t lTotalSendSize, int64_t rTotalSendSize)
      : _lBufferStart(std::move(lBufferStart)),
        _lBufferSize(std::move(lBufferSize)),
        _rBufferStart(std::move(rBufferStart)),
        _rBufferSize(std::move(rBufferSize)),
        _lRecvBufferSize(std::move(lRecvBufferSize)),
        _rRecvBufferSize(std::move(rRecvBufferSize)),
        _lSendSize(std::move(lSendSize)), _rSendSize(std::move(rSendSize)),
        _lSendOff(std::move(lSendOff)), _rSendOff(std::move(rSendOff)),
        _lRecvSize(std::move(lRecvSize)), _rRecvSize(std::move(rRecvSize)),
        _lRecvOff(std::move(lRecvOff)), _rRecvOff(std::move(rRecvOff)),
        _recvLBuff(std::move(recvLBuff)), _recvRBuff(std::move(recvRBuff)),
        _sendLBuff(std::move(sendLBuff)), _sendRBuff(std::move(sendRBuff)),
        _bufferizeSend(bufferizeSend), _bufferizeLRecv(bufferizeLRecv),
        _bufferizeRRecv(bufferizeRRecv), _lTotalRecvSize(lTotalRecvSize),
        _rTotalRecvSize(rTotalRecvSize), _lTotalSendSize(lTotalSendSize),
        _rTotalSendSize(rTotalSendSize) {}
  UHCache &operator=(const UHCache &) = delete;
  UHCache &operator=(UHCache &&) = default;
};

UHCache getMetaData(SHARPY::rank_type nworkers, int64_t ndims,
                    int64_t *ownedOff, int64_t *ownedShape,
                    int64_t *ownedStride, int64_t *bbOff, int64_t *bbShape,
                    int64_t *leftHaloShape, int64_t *leftHaloStride,
                    int64_t *rightHaloShape, int64_t *rightHaloStride,
                    SHARPY::Transceiver *tc) {
  UHCache cE; // holds data if non-cached
  auto myWorkerIndex = tc->rank();
  cE._lTotalRecvSize = 0;
  cE._rTotalRecvSize = 0;
  cE._lTotalSendSize = 0;
  cE._rTotalSendSize = 0;

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
  for (auto i = 0ul; i < nworkers; ++i) {
    offsets[i] = 2 * ndims * i;
  }
  tc->gather(bbTable.data(), counts.data(), offsets.data(), SHARPY::INT64,
             SHARPY::REPLICATED);

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
  cE._lSendOff.resize(nworkers, 0);
  cE._rSendOff.resize(nworkers, 0);
  cE._lSendSize.resize(nworkers, 0);
  cE._rSendSize.resize(nworkers, 0);

  // use send buffer if owned data is strided or sending a subview
  cE._bufferizeSend = (!SHARPY::is_contiguous(ownedShape, ownedStride, ndims) ||
                       bbTotCols != ownedTotCols);

  cE._lBufferStart.resize(nworkers * ndims, 0);
  cE._lBufferSize.resize(nworkers * ndims, 0);
  cE._rBufferStart.resize(nworkers * ndims, 0);
  cE._rBufferSize.resize(nworkers * ndims, 0);

  for (auto i = 0ul; i < nworkers; ++i) {
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
        if (cE._bufferizeSend) {
          cE._rSendOff[i] = i ? cE._rSendOff[i - 1] + cE._rSendSize[i - 1] : 0;
          cE._rBufferStart[i * ndims] = localRowStart;
          cE._rBufferSize[i * ndims] = nRows;
          for (auto j = 1; j < ndims; ++j) {
            cE._rBufferStart[i * ndims + j] = bbOff[j];
            cE._rBufferSize[i * ndims + j] = bbShape[j];
          }
        } else {
          cE._rSendOff[i] = localStart;
        }
        cE._rSendSize[i] = nSend;
        cE._rTotalSendSize += nSend;
      } else {
        // target is leftHalo
        if (cE._bufferizeSend) {
          cE._lSendOff[i] = i ? cE._lSendOff[i - 1] + cE._lSendSize[i - 1] : 0;
          cE._lBufferStart[i * ndims] = localRowStart;
          cE._lBufferSize[i * ndims] = nRows;
          for (auto j = 1; j < ndims; ++j) {
            cE._lBufferStart[i * ndims + j] = bbOff[j];
            cE._lBufferSize[i * ndims + j] = bbShape[j];
          }
        } else {
          cE._lSendOff[i] = localStart;
        }
        cE._lSendSize[i] = nSend;
        cE._lTotalSendSize += nSend;
      }
    }
  }

  // receive maps
  cE._lRecvSize.resize(nworkers);
  cE._rRecvSize.resize(nworkers);
  cE._lRecvOff.resize(nworkers);
  cE._rRecvOff.resize(nworkers);

  // receive size is sender's send size
  tc->alltoall(cE._lSendSize.data(), 1, SHARPY::INT32, cE._lRecvSize.data());
  tc->alltoall(cE._rSendSize.data(), 1, SHARPY::INT32, cE._rRecvSize.data());
  // compute offset in a contiguous receive buffer
  cE._lRecvOff[0] = 0;
  cE._rRecvOff[0] = 0;
  for (auto i = 1ul; i < nworkers; ++i) {
    cE._lRecvOff[i] = cE._lRecvOff[i - 1] + cE._lRecvSize[i - 1];
    cE._rRecvOff[i] = cE._rRecvOff[i - 1] + cE._rRecvSize[i - 1];
  }

  // receive buffering
  cE._bufferizeLRecv =
      !SHARPY::is_contiguous(leftHaloShape, leftHaloStride, ndims);
  cE._bufferizeRRecv =
      !SHARPY::is_contiguous(rightHaloShape, rightHaloStride, ndims);
  cE._lRecvBufferSize.resize(nworkers * ndims, 0);
  cE._rRecvBufferSize.resize(nworkers * ndims, 0);

  // deduce receive shape for unpack
  for (auto i = 0ul; i < nworkers; ++i) {
    if (cE._bufferizeLRecv && cE._lRecvSize[i] != 0) {
      cE._lTotalRecvSize += cE._lRecvSize[i];
      cE._lRecvBufferSize[i * ndims] = cE._lRecvSize[i] / bbTotCols; // nrows
      for (auto j = 1; j < ndims; ++j) {
        cE._lRecvBufferSize[i * ndims + j] = bbShape[j]; // leftHaloShape[j]
      }
    }
    if (cE._bufferizeRRecv && cE._rRecvSize[i] != 0) {
      cE._rTotalRecvSize += cE._rRecvSize[i];
      cE._rRecvBufferSize[i * ndims] = cE._rRecvSize[i] / bbTotCols; // nrows
      for (auto j = 1; j < ndims; ++j) {
        cE._rRecvBufferSize[i * ndims + j] = bbShape[j]; // rightHaloShape[j]
      }
    }
  }
  return cE;
};

/// @brief Update data in halo parts
/// We assume array is partitioned along the first dimension only
/// (row partitioning) and partitions are ordered by ranks
/// if cache-key is provided (>=0) meta data is read from cache
/// @return (MPI) handles
void *_idtr_update_halo(SHARPY::DTypeId sharpytype, int64_t ndims,
                        int64_t *ownedOff, int64_t *ownedShape,
                        int64_t *ownedStride, int64_t *bbOff, int64_t *bbShape,
                        void *ownedData, int64_t *leftHaloShape,
                        int64_t *leftHaloStride, void *leftHaloData,
                        int64_t *rightHaloShape, int64_t *rightHaloStride,
                        void *rightHaloData, SHARPY::Transceiver *tc,
                        int64_t key) {

#ifdef NO_TRANSCEIVER
  initMPIRuntime();
  tc = SHARPY::getTransceiver();
#endif

  if (!ownedOff || !ownedShape || !ownedStride || !bbOff || !bbShape ||
      !ownedData || !leftHaloShape || !leftHaloStride || !leftHaloData ||
      !rightHaloShape || !rightHaloStride || !rightHaloData || !tc) {
    throw std::runtime_error("Fatal error: received nullptr in update_halo.");
  }

  auto nworkers = tc->nranks();
  if (nworkers <= 1 || getenv("SHARPY_SKIP_COMM"))
    return nullptr;

  // not thread-safe
  static std::unordered_map<int64_t, UHCache> uhCache; // meta-data cache
  static UHCache *cache = nullptr; // reading either from non-cached or cached

  auto cIt = key == -1 ? uhCache.end() : uhCache.find(key);
  if (cIt == uhCache.end()) { // not in cache
    // update cache if requested
    cIt = uhCache
              .insert_or_assign(
                  key, std::move(getMetaData(
                           nworkers, ndims, ownedOff, ownedShape, ownedStride,
                           bbOff, bbShape, leftHaloShape, leftHaloStride,
                           rightHaloShape, rightHaloStride, tc)))
              .first;
  }
  cache = &(cIt->second);

  auto nbytes = sizeof_dtype(sharpytype);
  if (cache->_bufferizeLRecv) {
    cache->_recvLBuff.resize(cache->_lTotalRecvSize * nbytes);
  }
  if (cache->_bufferizeRRecv) {
    cache->_recvRBuff.resize(cache->_rTotalRecvSize * nbytes);
  }
  if (cache->_bufferizeSend) {
    cache->_sendLBuff.resize(cache->_lTotalSendSize * nbytes);
    cache->_sendRBuff.resize(cache->_rTotalSendSize * nbytes);
  }

  void *lRecvData =
      cache->_bufferizeLRecv ? cache->_recvLBuff.data() : leftHaloData;
  void *rRecvData =
      cache->_bufferizeRRecv ? cache->_recvRBuff.data() : rightHaloData;
  void *lSendData =
      cache->_bufferizeSend ? cache->_sendLBuff.data() : ownedData;
  void *rSendData =
      cache->_bufferizeSend ? cache->_sendRBuff.data() : ownedData;

  // communicate left/right halos
  if (cache->_bufferizeSend) {
    bufferize(ownedData, sharpytype, ownedShape, ownedStride,
              cache->_lBufferStart.data(), cache->_lBufferSize.data(), ndims,
              nworkers, cache->_sendLBuff.data());
  }
  auto lwh = tc->alltoall(lSendData, cache->_lSendSize.data(),
                          cache->_lSendOff.data(), sharpytype, lRecvData,
                          cache->_lRecvSize.data(), cache->_lRecvOff.data());
  if (cache->_bufferizeSend) {
    bufferize(ownedData, sharpytype, ownedShape, ownedStride,
              cache->_rBufferStart.data(), cache->_rBufferSize.data(), ndims,
              nworkers, cache->_sendRBuff.data());
  }
  auto rwh = tc->alltoall(rSendData, cache->_rSendSize.data(),
                          cache->_rSendOff.data(), sharpytype, rRecvData,
                          cache->_rRecvSize.data(), cache->_rRecvOff.data());

  auto wait = [=]() {
    tc->wait(lwh);
    std::vector<int64_t> recvBufferStart(nworkers * ndims, 0);
    if (cache->_bufferizeLRecv) {
      unpack(lRecvData, sharpytype, leftHaloShape, leftHaloStride,
             recvBufferStart.data(), cache->_lRecvBufferSize.data(), ndims,
             nworkers, leftHaloData);
    }
    tc->wait(rwh);
    if (cache->_bufferizeRRecv) {
      unpack(rRecvData, sharpytype, rightHaloShape, rightHaloStride,
             recvBufferStart.data(), cache->_rRecvBufferSize.data(), ndims,
             nworkers, rightHaloData);
    }
  };

  if (cache->_bufferizeLRecv || cache->_bufferizeRRecv ||
      getenv("SHARPY_NO_ASYNC")) {
    wait();
    return nullptr;
  }
  return mkWaitHandle(wait);
}

/// @brief templated wrapper for typed function versions calling
/// _idtr_update_halo
template <typename T>
void *_idtr_update_halo(SHARPY::Transceiver *tc, int64_t gShapeRank,
                        void *gShapeDescr, int64_t oOffRank, void *oOffDescr,
                        int64_t oDataRank, void *oDataDescr, int64_t bbOffRank,
                        void *bbOffDescr, int64_t bbShapeRank,
                        void *bbShapeDescr, int64_t lHaloRank, void *lHaloDescr,
                        int64_t rHaloRank, void *rHaloDescr, int64_t key) {

  if (!gShapeDescr || !oOffDescr || !oDataDescr || !bbOffDescr ||
      !bbShapeDescr || !lHaloDescr || !rHaloDescr) {
    throw std::runtime_error("Fatal error: received nullptr in update_halo.");
  }

  auto sharpytype = SHARPY::DTYPE<T>::value;

  // Construct unranked memrefs for metadata and data
  MRIdx1d ownedOff(oOffRank, oOffDescr);
  MRIdx1d bbOff(bbOffRank, bbOffDescr);
  MRIdx1d bbShape(bbShapeRank, bbShapeDescr);
  SHARPY::UnrankedMemRefType<T> ownedData(oDataRank, oDataDescr);
  SHARPY::UnrankedMemRefType<T> leftHalo(lHaloRank, lHaloDescr);
  SHARPY::UnrankedMemRefType<T> rightHalo(rHaloRank, rHaloDescr);

  return _idtr_update_halo(
      sharpytype, ownedData.rank(), ownedOff.data(), ownedData.sizes(),
      ownedData.strides(), bbOff.data(), bbShape.data(), ownedData.data(),
      leftHalo.sizes(), leftHalo.strides(), leftHalo.data(), rightHalo.sizes(),
      rightHalo.strides(), rightHalo.data(), tc, key);
}

extern "C" {
#define TYPED_UPDATE_HALO(_sfx, _typ)                                          \
  void *_idtr_update_halo_##_sfx(                                              \
      SHARPY::Transceiver *tc, int64_t gShapeRank, void *gShapeDescr,          \
      int64_t oOffRank, void *oOffDescr, int64_t oDataRank, void *oDataDescr,  \
      int64_t bbOffRank, void *bbOffDescr, int64_t bbShapeRank,                \
      void *bbShapeDescr, int64_t lHaloRank, void *lHaloDescr,                 \
      int64_t rHaloRank, void *rHaloDescr, int64_t key) {                      \
    return _idtr_update_halo<_typ>(                                            \
        tc, gShapeRank, gShapeDescr, oOffRank, oOffDescr, oDataRank,           \
        oDataDescr, bbOffRank, bbOffDescr, bbShapeRank, bbShapeDescr,          \
        lHaloRank, lHaloDescr, rHaloRank, rHaloDescr, key);                    \
  }                                                                            \
  _Pragma(STRINGIFY(weak _mlir_ciface__idtr_update_halo_##_sfx =               \
                        _idtr_update_halo_##_sfx))

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
