// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "CppTypes.hpp"

namespace SHARPY {

class NDSlice;

/// Futures and promises readily provide meta information
///   - id
///   - dtype (element type)
///   - shape (dim is unknown if < 0)
///   - device
///   - team
class ArrayMeta {
protected:
  id_type _guid = -1;
  DTypeId _dtype = DTYPE_LAST;
  shape_type _shape = {};
  std::string _device = {};
  std::string _team = 0;

public:
  ArrayMeta(id_type id, DTypeId dt, const shape_type &shape,
            const std::string &device, const std::string &team)
      : _guid(id), _dtype(dt), _shape(shape), _device(device), _team(team) {}
  ArrayMeta(id_type id, DTypeId dt, shape_type &&shape, std::string &&device,
            std::string &&team)
      : _guid(id), _dtype(dt), _shape(std::forward<shape_type>(shape)),
        _device(std::forward<std::string>(device)),
        _team(std::forward<std::string>(team)) {}
  ArrayMeta() = default;

  /// @return globally unique id
  id_type guid() const { return _guid; }

  /// @return dtype of future array
  DTypeId dtype() const { return _dtype; }

  /// @return shape of future array (dim is unknown if < 0)
  const shape_type &shape() const { return _shape; }

  /// @return rank of array (num of dims)
  rank_type rank() const { return _shape.size(); }

  // @ return device string, empty string means default
  const std::string &device() const { return _device; }

  // @ return team, 0 means non-distributed
  const std::string &team() const { return _team; }

  void set_guid(id_type guid) { _guid = guid; }
};

///
/// Abstract interface for a array implementation.
/// Used to hide the element type so we can bridge dynamic array types in Python
/// to C++.
///
class array_i {
public:
  /// (shared) pointer to a array
  typedef std::shared_ptr<array_i> ptr_type;

  /// Future array
  /// in addition to allowing getting value (get()).
  /// TFuture also readily provides meta information
  ///   - id
  ///   - dtype (element type)
  ///   - shape
  ///   - device and team
  template <typename T> class Metaified : public T, public ArrayMeta {
  public:
    Metaified() = default;
    Metaified(T &&f, id_type id, DTypeId dt, shape_type &&shape,
              std::string &&device, std::string &&team)
        : T(std::move(f)), ArrayMeta(id, dt, std::move(shape),
                                     std::move(device), std::move(team)) {}
    Metaified(T &&f, id_type id, DTypeId dt, const shape_type &shape,
              const std::string &device, const std::string &team)
        : T(std::move(f)), ArrayMeta(id, dt, shape, device, team) {}
    Metaified(id_type id, DTypeId dt, const shape_type &shape,
              const std::string &device, const std::string &team)
        : T(), ArrayMeta(id, dt, shape, device, team) {}
    Metaified(id_type id, DTypeId dt, shape_type &&shape, std::string &&device,
              std::string &&team)
        : T(), ArrayMeta(id, dt, std::forward<shape_type>(shape),
                         std::move(device), std::move(team)) {}
    ~Metaified() {}
  };

  /// promise type producing a future array
  using promise_type = Metaified<std::promise<ptr_type>>;
  using future_type = Metaified<std::shared_future<array_i::ptr_type>>;

  virtual ~array_i() {};
  /// python object's __repr__
  virtual std::string __repr__() const = 0;
  /// @return array's element type
  virtual DTypeId dtype() const = 0;
  /// @return array's shape
  virtual const shape_type &shape() const = 0;
  /// @return number of dimensions of array
  virtual int ndims() const = 0;
  /// @return global number of elements in array
  virtual uint64_t size() const = 0;
  /// @return boolean value of 0d array
  virtual bool __bool__() const = 0;
  /// @return float value of 0d array
  virtual double __float__() const = 0;
  /// @return integer value of 0d array
  virtual int64_t __int__() const = 0;
  /// @return global number of elements in first dimension
  virtual uint64_t __len__() const = 0;

  // size of a single element (in bytes)
  virtual int item_size() const = 0;
  // mark as deallocated
  virtual void markDeallocated() = 0;
  virtual bool isAllocated() = 0;
};

#if 0
template<typename S>
void serialize(S & ser, array_i::future_type & f)
{
    uint64_t id = f.guid();
    ser.value8b(id);
    if constexpr (std::is_same<Deserializer, S>::value) {

    }
}
#endif
} // namespace SHARPY
