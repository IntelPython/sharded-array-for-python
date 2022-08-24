// SPDX-License-Identifier: BSD-3-Clause

#include "include/ddptensor/Deferred.hpp"
#include "include/ddptensor/Transceiver.hpp"
#include "include/ddptensor/Mediator.hpp"
#include "include/ddptensor/Registry.hpp"

#include <oneapi/tbb/concurrent_queue.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <iostream>
#include <unordered_set>

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
    return {std::move(promise_type::get_future().share()), _guid, _dtype, _rank};
}

Deferred::future_type defer_tensor(Runable::ptr_type && _d, bool is_global)
{
    Deferred * d = dynamic_cast<Deferred*>(_d.get());
    if(!d) throw std::runtime_error("Expected Deferred Tensor promise");
    if(is_global) {
        _dist(d);
        d->set_guid(Registry::get_guid());
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

#if 0
class DepManager
{
private:
    IdValueMap _ivm;
    std::unordered_set<id_type> _args;
public:
    ::mlir::Value getDependent(i::mlir::OpBuilder & builder, d_type guid)
    {
        if(auto d = _ivm.find(guid); d == _ivm.end()) {
            _func.insertArg
            _ivm[guid] = {val, {}}
        } else {
            return d->second.first;
        }
    }
};
#endif

void process_promises()
{
    bool done = false;
    jit::JIT jit;

    do {
        ::mlir::OpBuilder builder(&jit._context);
        auto loc = builder.getUnknownLoc();
        jit::IdValueMap ivp;
    
        // Create a MLIR module
        auto module = builder.create<::mlir::ModuleOp>(loc);
        // Create a func
        auto dtype = builder.getI64Type();
        // create dummy type, we'll replace it with the actual type later
        auto dummyFuncType = builder.getFunctionType({}, dtype);
        std::string fname("ddpt_jit");
        auto function = builder.create<::mlir::func::FuncOp>(loc, fname, dummyFuncType);
        // create function entry block
        auto &entryBlock = *function.addEntryBlock();
        // Set the insertion point in the builder to the beginning of the function body
        builder.setInsertionPointToStart(&entryBlock);
        // we need to keep runables/deferred/futures alive until we set their values below
        std::vector<Runable::ptr_type> runables;

        while(true) {
            Runable::ptr_type d;
            _deferred.pop(d);
            if(d) {
                if(d->generate_mlir(builder, loc, ivp)) {
                    d.reset();
                    break;
                };
                // keep alive for later set_value
                runables.push_back(std::move(d));
            } else {
                // signals system shutdown
                done = true;
                break;
            }
        }

        if(runables.empty()) continue;

        // Now we have to define the return type as a ValueRange of all arrays which we have created
        // (runnables have put them into ivp when generating mlir)
        // We also compute the total size of the struct llvm created for this return type
        // llvm will basically return a struct with all the arrays as members, each of type JIT::MemRefDescriptor

        // Need a container to put all return values, will be used to construct TypeRange
        std::vector<::mlir::Type> ret_types;
        ret_types.reserve(ivp.size());
        std::vector<::mlir::Value> ret_values;
        ret_types.reserve(ivp.size());
        std::unordered_map<id_type, uint64_t> rank_map;
        // here we store the total size of the llvm struct
        uint64_t sz = 0;
        for(auto & v : ivp) {
            auto value = v.second.first;
            // append the type and array/value
            ret_types.push_back(value.getType());
            ret_values.push_back(value);
            auto ptt = value.getType().dyn_cast<::imex::ptensor::PTensorType>();
            assert(ptt);
            auto rank = ptt.getRtensor().getShape().size();
            rank_map[v.first] = rank;
            // add sizeof(MemRefDescriptor<elementtype, rank>) to sz
            sz += 3 + 2 * rank;
        }
        ::mlir::TypeRange ret_tr(ret_types);
        ::mlir::ValueRange ret_vr(ret_values);

        // add return statement
        auto ret_value = builder.create<::mlir::func::ReturnOp>(loc, ret_vr);
        // Define and assign correct function type
        auto funcTypeAttr = ::mlir::TypeAttr::get(builder.getFunctionType({}, ret_tr));
        function.setFunctionTypeAttr(funcTypeAttr);
        // also request generation of c-wrapper function
        function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(), ::mlir::UnitAttr::get(&jit._context));
        // add the function to the module
        module.push_back(function);
        module.dump();

        // compile and run the module
        assert(sizeof(intptr_t) == sizeof(void*));
        intptr_t * output = new intptr_t[sz];
        std::cout << ivp.size() << " sz: " << sz << std::endl;
        if(jit.run(module, fname, output)) throw std::runtime_error("failed running jit");

        // push results to fulfill promises
        size_t pos = 0;
        for(auto & v : ivp) {
            auto value = v.second.first;
            auto rank = rank_map[v.first];
            void * allocated = (void*)output[pos];
            void * aligned = (void*)output[pos+1];
            intptr_t offset = output[pos+2];
            intptr_t * sizes = output + pos + 3;
            intptr_t * stride = output + pos + 3 + rank;
            pos += 3 + 2 * rank;
            v.second.second(rank, allocated, aligned, offset, sizes, stride);
        }
    } while(!done);
}

void sync()
{
    // FIXME this does not wait for the last deferred to complete
    while(!_deferred.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

