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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace SHARPY {

NDArray::NDArray(id_type guid_, DTypeId dtype_, shape_type gShape,
                 const std::string &device_, const std::string &team_,
                 void *allocated, void *aligned, intptr_t offset,
                 const intptr_t *sizes, const intptr_t *strides,
                 std::vector<int64_t> &&loffs, rank_type owner)
    : ArrayMeta(guid_, dtype_, gShape, device_, team_), _owner(owner),
      _lData(allocated ? gShape.size() : 0, allocated, aligned, offset, sizes,
             strides),
      _lOffsets(std::move(loffs)) {
  if (ndims() == 0) {
    _owner = REPLICATED;
  }
}

NDArray::NDArray(id_type guid_, DTypeId dtype_, const shape_type &shp,
                 const std::string &device_, const std::string &team_,
                 rank_type owner)
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
  delete[] sizes;
  delete[] strides;
}

// incomplete, useful for computing meta information
NDArray::NDArray(id_type guid_, const int64_t *shape, uint64_t N,
                 const std::string &device_, const std::string &team_,
                 rank_type owner)
    : ArrayMeta(guid_, DTYPE_LAST, {shape, shape + N}, device_, team_),
      _owner(owner) {
  assert(ndims() <= 1);
}

// from numpy
NDArray::NDArray(id_type guid_, DTypeId dtype_, ssize_t ndims,
                 const ssize_t *shape, const intptr_t *strides, void *data,
                 const std::string &device_, const std::string &team_)
    : ArrayMeta(guid_, dtype_, {shape, shape + ndims}, device_, team_),
      _owner(NOOWNER),
      _lData(ndims, data, data, 0, reinterpret_cast<const intptr_t *>(shape),
             reinterpret_cast<const intptr_t *>(strides)),
      _lOffsets(ndims, 0) {}

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
          // don't do anything if runtime was shutdown
          if (finied) {
            std::cerr << "sharpy fini: detected possible memory leak\n";
          } else {
            auto av = dm.addDependent(builder, a);
            auto deleteOp = builder.create<::imex::ndarray::DeleteOp>(loc, av);
            deleteOp->setAttr("bufferization.manual_deallocation",
                              builder.getUnitAttr());
            dm.drop(a->guid());
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
  if (_base) {
    if (_base->needGIL()) {
      py::gil_scoped_acquire acquire;
      delete _base;
    } else {
      delete _base;
    }
  }
}

// **************************************************************************

bool NDArray::isAllocated() { return !_base && _lData._allocated != nullptr; }

void NDArray::markDeallocated() { _lData.markDeallocated(); }

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
  if (_lOffsets.size())
    for (auto i = 0; i < nd; ++i)
      oss << _lOffsets[i] << (i == nd - 1 ? "" : ", ");
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
