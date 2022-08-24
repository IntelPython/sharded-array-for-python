// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"
#include "Registry.hpp"
#include "jit/mlir.hpp"

extern void process_promises();
extern void sync();

struct Runable
{
    using ptr_type = std::unique_ptr<Runable>;
    virtual ~Runable() {};
    /// actually execute, a deferred will set value of future
    virtual void run() = 0;
    /// generate MLIR code for jit
    /// @return true if last operation in to-be-compiled region, false otherwise
    virtual bool generate_mlir(::mlir::OpBuilder &, ::mlir::Location, jit::IdValueMap &)
    {
        throw(std::runtime_error("No MLIR support for this operation."));
        return false;
    };
    virtual FactoryId factory() const = 0;
    virtual void defer(ptr_type &&);
    static void fini();
};

extern void push_runable(Runable::ptr_type && r);

template<typename P, typename F>
struct DeferredT : public P, public Runable
{
    using ptr_type = std::unique_ptr<DeferredT>;
    using promise_type = P;
    using future_type = F;

    DeferredT() = default;
    DeferredT(const DeferredT<P, F> &) = delete;
};

/// Deferred operation returning/producing a tensor
/// holds a guid as well as rank and dtype of future tensor
class Deferred : public DeferredT<tensor_i::promise_type, tensor_i::future_type>
{
public:
    using ptr_type = std::unique_ptr<Deferred>;
    
    Deferred(DTypeId dt, int rank)
    : _guid(Registry::NOGUID), // might be set later
      _dtype(dt),
      _rank(rank)
    {}
    Deferred(id_type guid, DTypeId dt, int rank)
    : _guid(guid),
      _dtype(dt),
      _rank(rank)
    {}
    // FIXME we should not allow default values for dtype and rank
    // we should need this only while we are gradually moving to mlir
    Deferred()
    : _guid(Registry::NOGUID),
      _dtype(DTYPE_LAST),
      _rank(-1)
    {}

    id_type guid() const { return _guid; }
    DTypeId dtype() const { return _dtype; }
    int rank() const { return _rank; }

    void set_guid(id_type guid) {
        _guid = guid;
    }

    future_type get_future();
    // from Runable
    void defer(Runable::ptr_type &&);

protected:
    id_type _guid;
    DTypeId _dtype;
    int     _rank;
};

extern void _dist(const Runable * p);

// defer operations which do not return a tensor, e.g. which are not a Deferred
template<typename T, typename... Ts, std::enable_if_t<!std::is_base_of_v<Deferred, T>, bool> = true>
typename T::future_type defer(Ts&&... args)
{
    auto p = std::make_unique<T>(std::forward<Ts>(args)...);
    _dist(p.get());
    auto f = p->get_future().share();
    push_runable(std::move(p));
    return f;
}

// implementation details for deferring ops returning tensors 
extern Deferred::future_type defer_tensor(Runable::ptr_type && d, bool is_global);

// defer operations which do return a tensor, e.g. which are a Deferred
template<typename T, typename... Ts, std::enable_if_t<std::is_base_of_v<Deferred, T>, bool> = true>
Deferred::future_type defer(Ts&&... args)
{
    return defer_tensor(std::move(std::make_unique<T>(std::forward<Ts>(args)...)), true);
}

static void defer(nullptr_t)
{
    push_runable(Runable::ptr_type());
}

struct UnDeferred : public Deferred
{
    UnDeferred(tensor_i::ptr_type ptr)
    {
        set_value(std::move(ptr));
    }

    void run()
    {
    }

    FactoryId factory() const
    {
        throw(std::runtime_error("No Factory for Undeferred."));
    }
};

template<typename L>
struct DeferredLambda : public Runable
{
    using promise_type = int;
    using future_type = int;

    L _l;

    DeferredLambda(L l)
        : _l(l)
    {}

    void run()
    {
        _l();
    }

    bool generate_mlir(::mlir::OpBuilder &, ::mlir::Location, jit::IdValueMap &)
    {
        return _l();
    }

    FactoryId factory() const
    {
        throw(std::runtime_error("No Factory for DeferredLambda."));
    }
};

template<typename L>
void defer_lambda(L && l)
{
    push_runable(std::move(std::make_unique<DeferredLambda<L>>(l)));
}
