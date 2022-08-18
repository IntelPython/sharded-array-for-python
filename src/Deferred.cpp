// SPDX-License-Identifier: BSD-3-Clause

#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Transceiver.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"

#include <oneapi/tbb/concurrent_queue.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

static tbb::concurrent_bounded_queue<Runable::ptr_type> _deferred;

void push_runable(Runable::ptr_type && r)
{
    _deferred.push(std::move(r));
}

void _dist(const Runable * p)
{
    if(is_cw() && theTransceiver->rank() == 0)
        theMediator->to_workers(p);
}

Deferred::future_type Deferred::get_future()
{
    return {std::move(tensor_i::promise_type::get_future().share()), _guid};
}

Deferred::future_type defer_tensor(Runable::ptr_type && _d, bool is_global)
{
    Deferred * d = dynamic_cast<Deferred*>(_d.get());
    if(!d) throw std::runtime_error("Expected Deferred Tensor promise");
    if(is_global) {
        _dist(d);
        d->_guid = Registry::get_guid();
    }
    auto f = d->get_future();
    Registry::put(f);
    push_runable(std::move(_d));
    return f;
}

void Deferred::defer(Runable::ptr_type && p)
{
    defer_tensor(std::move(p), true);
}

void Runable::defer(Runable::ptr_type && p)
{
    push_runable(std::move(p));
}

void Runable::fini()
{
    _deferred.clear();
}

void process_promises()
{
    jit::JIT jit;
    ::mlir::OpBuilder builder(&jit._context);
    auto loc = builder.getUnknownLoc();
    jit::IdValueMap ivp;
    ::mlir::Value ret_value;

    // Create a MLIR module
    auto module = builder.create<::mlir::ModuleOp>(loc);
    // Create a func
    auto dtype = builder.getI64Type();
    llvm::SmallVector<int64_t> shape(1, -1); //::mlir::ShapedType::kDynamicSize);
    auto rrtype = ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(shape, dtype), true); // llvm::SmallVector<int64_t>()
    auto funcType = builder.getFunctionType({}, rrtype);
    std::string fname("ddpt_jit");
    auto function = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
    // request generation of c-wrapper function
    function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(), ::mlir::UnitAttr::get(&jit._context));
    // create function entry block
    auto &entryBlock = *function.addEntryBlock();
    // Set the insertion point in the builder to the beginning of the function body
    builder.setInsertionPointToStart(&entryBlock);
    
    while(true) {
        Runable::ptr_type d;
        _deferred.pop(d);
        if(d) {
            d->run();
            ret_value = d->generate_mlir(builder, loc, ivp);
            d.reset();
        } else {
            break;
        }
    }

    (void)builder.create<::mlir::func::ReturnOp>(loc, ret_value);
    // add the function to the module
    module.push_back(function);
    module.dump();
    // finally compile and run the module
    if(jit.run(module, fname)) throw std::runtime_error("failed running jit");
}

void sync()
{
    // FIXME this does not wait for the last deferred to complete
    while(!_deferred.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

