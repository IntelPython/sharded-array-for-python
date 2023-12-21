// SPDX-License-Identifier: BSD-3-Clause

// Concrete implementation of array_i.
// Interfaces are based on shared_ptr<array_i>.

#include <sharpy/CppTypes.hpp>
#include <sharpy/Deferred.hpp>
#include <sharpy/NDArray.hpp>
#include <sharpy/Transceiver.hpp>
#include <sharpy/jit/mlir.hpp>

#include <algorithm>
#include <iostream>

namespace SHARPY {

NDArray::NDArray(id_type guid_, DTypeId dtype_, shape_type gShape,
                 std::string device_, uint64_t team_, void *l_allocated,
                 void *l_aligned, intptr_t l_offset, const intptr_t *l_sizes,
                 const intptr_t *l_strides, void *o_allocated, void *o_aligned,
                 intptr_t o_offset, const intptr_t *o_sizes,
                 const intptr_t *o_strides, void *r_allocated, void *r_aligned,
                 intptr_t r_offset, const intptr_t *r_sizes,
                 const intptr_t *r_strides, uint64_t *lo_allocated,
                 uint64_t *lo_aligned, rank_type owner)
    : ArrayMeta(guid_, dtype_, gShape, device_, team_), _owner(owner),
      _lo_allocated(lo_allocated), _lo_aligned(lo_aligned),
      _lhsHalo(l_allocated ? gShape.size() : 0, l_allocated, l_aligned,
               l_offset, l_sizes, l_strides),
      _lData(o_allocated ? gShape.size() : 0, o_allocated, o_aligned, o_offset,
             o_sizes, o_strides),
      _rhsHalo(r_allocated ? gShape.size() : 0, r_allocated, r_aligned,
               r_offset, r_sizes, r_strides) {
  if (ndims() == 0) {
    _owner = REPLICATED;
  }
  assert(team() == 0 || transceiver() == getTransceiver());
}

NDArray::NDArray(id_type guid_, DTypeId dtype_, const shape_type &shp,
                 std::string device_, uint64_t team_, rank_type owner)
    : ArrayMeta(guid_, dtype_, shp, device_, team_), _owner(owner) {

  auto esz = sizeof_dtype(dtype_);
  auto lsz =
      std::accumulate(shp.begin(), shp.end(), esz, std::multiplies<intptr_t>());
  auto allocated = aligned_alloc(esz, lsz);
  auto nds = ndims();
  auto sizes = new intptr_t[nds];
  auto strides = new intptr_t[nds];
  intptr_t stride = 1;
  assert(nds <= 1);
  for (auto i = 0; i < nds; ++i) {
    sizes[i] = shp[i];
    strides[nds - i - 1] = stride;
    stride *= shp[i];
  }
  _lData = DynMemRef(nds, allocated, allocated, 0, sizes, strides);
  assert(team() == 0 || transceiver() == getTransceiver());
}

// incomplete, useful for computing meta information
NDArray::NDArray(id_type guid_, const int64_t *shape, uint64_t N,
                 std::string device_, uint64_t team_, rank_type owner)
    : ArrayMeta(guid_, DTYPE_LAST, {shape, shape + N}, device_, team_),
      _owner(owner) {
  assert(ndims() <= 1);
  assert(team() == 0 || transceiver() == getTransceiver());
}

// from numpy
NDArray::NDArray(id_type guid_, DTypeId dtype_, ssize_t ndims,
                 const ssize_t *shape, const intptr_t *strides, void *data,
                 std::string device_, uint64_t team_)
    : ArrayMeta(guid_, dtype_, {shape, shape + ndims}, device_, team_),
      _owner(NOOWNER), _lo_allocated(static_cast<uint64_t *>(
                           calloc(ndims, sizeof_dtype(dtype_)))),
      _lo_aligned(_lo_allocated),
      _lData(ndims, data, data, 0, reinterpret_cast<const intptr_t *>(shape),
             reinterpret_cast<const intptr_t *>(strides)) {}

void NDArray::set_base(const array_i::ptr_type &base) {
  _base = new SharedBaseObject<array_i::ptr_type>(base);
}
void NDArray::set_base(BaseObj *obj) { _base = obj; }

// **************************************************************************

extern bool finied;

// NDArray's deleter makes the deallocation asynchronous. The whole processes is
// very sensitive, in particular the lifetime of the to-be-deleted array and its
// pointers needs to be handled with care. Generating MLIR requires the array to
// be alive and intact until the function was actually invoked (to extract the
// memrefs). Hence we deallocate as follows
// - create a deferred object which generates MLIR to free the array data
//   - it provides a callback to MLIR which is called after execution. this
//     deletes remaining heap allocations.
// - create a deferred which deletes the pointer itself (must go *after* the
//   first).
// - NDArray's destructor does not delete an memory except its base.
void NDArray::NDADeleter::operator()(NDArray *a) const {
  if (!a->_base && a->isAllocated()) {
    // create MLIR to deallocate as deferred
    defer_del_lambda(
        [a](::mlir::OpBuilder &builder, const ::mlir::Location &loc,
            jit::DepManager &dm) {
          assert(a);
          uint64_t *ptr = const_cast<uint64_t *>(a->_lo_allocated);
          // don't do anything if runtime was shutdown
          if (finied) {
            std::cerr << "sharpy fini: detected possible memory leak\n";
            if (ptr) {
              free(ptr);
            }
          } else {
            auto av = dm.getDependent(builder, a);
            builder.create<::imex::ndarray::DeleteOp>(loc, av);
            dm.drop(a->guid());

            if (ptr) {
              // further defer deleting remaining memory until after execution
              dm.addReady(a->guid(), [ptr](id_type guid) { free(ptr); });
            }
          }
          return false;
        },
        []() {});

    // actually delete pointer as a deferred to be executed *after* the above
    defer_del_lambda(
        [a](auto, auto, auto) {
          delete a;
          return false;
        },
        []() {});
  } else {
    delete a;
  }
}

NDArray::~NDArray() {
  if (_base)
    delete _base;
}

// **************************************************************************

bool NDArray::isAllocated() { return !_base && _lData._allocated != nullptr; }

void NDArray::markDeallocated() {
  _lhsHalo.markDeallocated();
  _lData.markDeallocated();
  _rhsHalo.markDeallocated();
}

void *NDArray::data() {
  void *ret;
  dispatch(dtype(), _lData._aligned,
           [this, &ret](auto *ptr) { ret = ptr + this->_lData._offset; });
  return ret;
}

bool NDArray::is_sliced() const {
  if (ndims() == 0)
    return false;
  auto d = ndims() - 1;
  intptr_t tsz = _lData._strides[d];
  if (tsz == 1) {
    for (; d > 0; --d) {
      tsz *= _lData._sizes[d];
      if (tsz <= 0)
        break;
      if (_lData._strides[d - 1] > tsz)
        return true;
    }
  }
  return false;
}

std::string NDArray::__repr__() const {
  const auto nd = ndims();
  std::ostringstream oss;
  oss << "ndarray{gs=(";
  auto gshp = ArrayMeta::shape();
  for (auto i = 0; i < nd; ++i)
    oss << gshp[i] << (i == nd - 1 ? "" : ", ");
  oss << "), loff=(";
  if (_lo_aligned)
    for (auto i = 0; i < nd; ++i)
      oss << _lo_aligned[i] << (i == nd - 1 ? "" : ", ");
  oss << "), lsz=(";
  for (auto i = 0; i < nd; ++i)
    oss << _lData._sizes[i] << (i == nd - 1 ? "" : ", ");
  oss << "), str=(";
  for (auto i = 0; i < nd; ++i)
    oss << _lData._strides[i] << (i == nd - 1 ? "" : ", ");
  oss << "), p=" << _lData._allocated << ", poff=" << _lData._offset
      << ", team=" << team() << "}\n";

  dispatch(dtype(), _lData._aligned, [this, nd, &oss](auto *ptr) {
    auto cptr = ptr + this->_lData._offset;
    if (nd > 0) {
      printit(oss, 0, cptr);
    } else {
      oss << *cptr;
    }
  });
  return oss.str();
}

bool NDArray::__bool__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar bool: array is not replicated"));

  bool res;
  dispatch(dtype(), _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<bool>(ptr[this->_lData._offset]);
  });
  return res;
}

double NDArray::__float__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar float: array is not replicated"));

  double res;
  dispatch(dtype(), _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<double>(ptr[this->_lData._offset]);
  });
  return res;
}

int64_t NDArray::__int__() const {
  if (!is_replicated())
    throw(std::runtime_error("Cast to scalar int: array is not replicated"));

  float res;
  dispatch(dtype(), _lData._aligned, [this, &res](auto *ptr) {
    res = static_cast<float>(ptr[this->_lData._offset]);
  });
  return res;
}

void NDArray::add_to_args(std::vector<void *> &args) const {
  int ndims = this->ndims();
  auto storeMR = [ndims](const DynMemRef &mr) -> intptr_t * {
    intptr_t *buff = new intptr_t[memref_sz(ndims)];
    buff[0] = reinterpret_cast<intptr_t>(mr._allocated);
    buff[1] = reinterpret_cast<intptr_t>(mr._aligned);
    buff[2] = static_cast<intptr_t>(mr._offset);
    memcpy(buff + 3, mr._sizes, ndims * sizeof(intptr_t));
    memcpy(buff + 3 + ndims, mr._strides, ndims * sizeof(intptr_t));
    return buff;
  }; // FIXME memory leak?

  if (team() == 0 || ndims == 0) {
    // no-dist-mode
    args.push_back(storeMR(_lData));
  } else {
    args.push_back(storeMR(_lhsHalo));
    args.push_back(storeMR(_lData));
    args.push_back(storeMR(_rhsHalo));
    // local offsets last
    auto buff = new intptr_t[memref_sz(1)];
    assert(5 == memref_sz(1));
    buff[0] = reinterpret_cast<intptr_t>(_lo_allocated);
    buff[1] = reinterpret_cast<intptr_t>(_lo_aligned);
    buff[2] = 0;
    buff[3] = ndims;
    buff[4] = 1;
    args.push_back(buff);
  }
}

void NDArray::replicate() {
  if (is_replicated())
    return;
  auto gsz = size();
  auto lsz = local_size();
  if (gsz > 1)
    throw(std::runtime_error(
        "Replication implemented for single-element arrays only."));
  if (lsz != gsz) {
    assert(lsz == 0);
    auto nd = ndims();
    for (auto i = 0; i < nd; ++i) {
      _lData._sizes[i] = _lData._strides[i] = 1;
    }
    _lData._sizes[nd - 1] = gsz;
  }
  dispatch(dtype(), _lData._aligned, [this, lsz, gsz](auto *ptr) {
    auto tmp = ptr[this->_lData._offset];
    if (lsz != gsz)
      ptr[this->_lData._offset] = 0;
    getTransceiver()->reduce_all(&ptr[this->_lData._offset], this->dtype(), 1,
                                 SUM);
    assert(lsz != gsz || tmp == ptr[this->_lData._offset]);
  });
  set_owner(REPLICATED);
}
} // namespace SHARPY
