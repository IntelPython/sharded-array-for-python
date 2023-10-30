// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "CppTypes.hpp"

namespace DDPT {

class NDSlice;

/// Futures and promises readily provide meta information
///   - id
///   - dtype (element type)
///   - shape (dim is unknown if < 0)
///   - team
///   - balanced (flag indicating if tensor's partitions are balanced)
class TensorMeta {
protected:
  id_type _guid = -1;
  DTypeId _dtype = DTYPE_LAST;
  shape_type _shape = {};
  uint64_t _team;
  bool _balanced = true;

public:
  TensorMeta(id_type id, DTypeId dt, const shape_type &shape, uint64_t team,
             bool balanced)
      : _guid(id), _dtype(dt), _shape(shape), _team(team), _balanced(balanced) {
  }
  TensorMeta(id_type id, DTypeId dt, shape_type &&shape, uint64_t team,
             bool balanced)
      : _guid(id), _dtype(dt), _shape(std::forward<shape_type>(shape)),
        _team(team), _balanced(balanced) {}
  TensorMeta() = default;

  /// @return globally unique id
  id_type guid() const { return _guid; }

  /// @return dtype of future tensor
  DTypeId dtype() const { return _dtype; }

  /// @return shape of future tensor (dim is unknown if < 0)
  const shape_type &shape() const { return _shape; }

  /// @return rank of tensor (num of dims)
  rank_type rank() const { return _shape.size(); }

  // @ return team, 0 means non-distributed
  uint64_t team() const { return _team; }

  /// @return if future tensor will have balanced partitions
  int balanced() const { return _balanced; }

  void set_guid(id_type guid) { _guid = guid; }
};

///
/// Abstract interface for a tensor implementation.
/// Used to hide the element type so we can bridge dynamic array types in Python
/// to C++.
///
class tensor_i {
public:
  /// (shared) pointer to a tensor
  typedef std::shared_ptr<tensor_i> ptr_type;

  /// Future tensor
  /// in addition to allowing getting value (get()).
  /// TFuture also readily provides meta information
  ///   - id
  ///   - dtype (element type)
  ///   - shape
  ///   - balanced (flag indicating if tensor's partitions are balanced)
  template <typename T> class Metaified : public T, public TensorMeta {
  public:
    Metaified() = default;
    Metaified(T &&f, id_type id, DTypeId dt, shape_type &&shape, uint64_t team,
              bool balanced)
        : T(std::move(f)),
          TensorMeta(id, dt, std::move(shape), team, balanced) {}
    Metaified(T &&f, id_type id, DTypeId dt, const shape_type &shape,
              uint64_t team, bool balanced)
        : T(std::move(f)), TensorMeta(id, dt, shape, team, balanced) {}
    Metaified(id_type id, DTypeId dt, const shape_type &shape, uint64_t team,
              bool balanced)
        : T(), TensorMeta(id, dt, shape, team, balanced) {}
    Metaified(id_type id, DTypeId dt, shape_type &&shape, uint64_t team,
              bool balanced)
        : T(),
          TensorMeta(id, dt, std::forward<shape_type>(shape), team, balanced) {}
    ~Metaified() {}
  };

  /// promise type producing a future tensor
  using promise_type = Metaified<std::promise<ptr_type>>;
  using future_type = Metaified<std::shared_future<tensor_i::ptr_type>>;

  virtual ~tensor_i(){};
  /// python object's __repr__
  virtual std::string __repr__() const = 0;
  /// @return tensor's element type
  virtual DTypeId dtype() const = 0;
  /// @return tensor's shape
  virtual const int64_t *shape() const = 0;
  /// @return number of dimensions of tensor
  virtual int ndims() const = 0;
  /// @return global number of elements in tensor
  virtual uint64_t size() const = 0;
  /// @return boolean value of 0d tensor
  virtual bool __bool__() const = 0;
  /// @return float value of 0d tensor
  virtual double __float__() const = 0;
  /// @return integer value of 0d tensor
  virtual int64_t __int__() const = 0;
  /// @return global number of elements in first dimension
  virtual uint64_t __len__() const = 0;

  // size of a single element (in bytes)
  virtual int item_size() const = 0;
  // store tensor information in form of corresponding
  // jit::JIT::DistMemRefDescriptor
  // @return stored size in number of intptr_t
  virtual void add_to_args(std::vector<void *> &args) = 0;
};

#if 0
template<typename S>
void serialize(S & ser, tensor_i::future_type & f)
{
    uint64_t id = f.guid();
    ser.value8b(id);
    if constexpr (std::is_same<Deserializer, S>::value) {

    }
}
#endif
} // namespace DDPT
