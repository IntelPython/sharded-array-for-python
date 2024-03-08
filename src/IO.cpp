// SPDX-License-Identifier: BSD-3-Clause

/*
  I/O ops.
*/

#include "sharpy/IO.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/NDArray.hpp"
#include "sharpy/PyTypes.hpp"
#include "sharpy/SetGetItem.hpp"
#include "sharpy/Transceiver.hpp"
#include "sharpy/TypeDispatch.hpp"

namespace SHARPY {

// ***************************************************************************

/// @brief form a FutureArray from local numpy arrays (inplace - no copy)
struct DeferredFromLocal : public Deferred {
  py::array _npa;

  DeferredFromLocal() = default;
  DeferredFromLocal(py::array npa)
      : Deferred(getDTypeId(npa.dtype()),
                 {npa.shape(), npa.shape() + npa.ndim()}, {}, 0),
        _npa(npa) {}

  // get our DTypeId from py::dtype
  DTypeId getDTypeId(const py::dtype &dtype) {
    auto bw = dtype.itemsize();
    auto kind = dtype.kind();
    switch (kind) {
    case 'i':
      switch (bw) {
      case 1:
        return INT8;
      case 2:
        return INT16;
      case 4:
        return INT32;
      case 8:
        return INT64;
      };
    case 'f':
      switch (bw) {
      case 4:
        return FLOAT32;
      case 8:
        return FLOAT64;
      };
    };
    throw std::runtime_error("Unsupported dtype");
  }

  void run() override {
    auto _strides = _npa.strides();
    auto shape = _npa.shape();
    auto data = _npa.mutable_data();
    auto dtype = _npa.dtype();
    auto ndim = _npa.ndim();
    auto eSz = dtype.itemsize();

    // py::array stores strides in bytes, not elements
    std::vector<intptr_t> strides(ndim);
    for (auto i = 0; i < ndim; ++i) {
      strides[i] = _strides[i] / eSz;
    }

    auto res = mk_tnsr(this->guid(), getDTypeId(dtype), ndim, shape,
                       strides.data(), data, this->device(), this->team());
    // make sure we do not delete numpy's memory before the numpy array is dead
    // notice: py::objects have ref-counting)
    res->set_base(new SharedBaseObject<py::object>(_npa));
    set_value(std::move(res));
  }

  bool generate_mlir(::mlir::OpBuilder &builder, const ::mlir::Location &loc,
                     jit::DepManager &dm) override {
    return true;
  }

  FactoryId factory() const override { return F_FROMLOCALS; }

  template <typename S> void serialize(S &ser) {}
};

GetItem::py_future_type IO::to_numpy(const FutureArray &a) {
  assert(!getTransceiver()->is_cw() || getTransceiver()->rank() == 0);
  return GetItem::gather(a, getTransceiver()->is_cw() ? 0 : REPLICATED);
}

FutureArray *IO::from_locals(const std::vector<py::array> &a) {
  assert(a.size() == 1);
  return new FutureArray(defer<DeferredFromLocal>(a.front()));
}

FACTORY_INIT(DeferredFromLocal, F_FROMLOCALS);
} // namespace SHARPY
