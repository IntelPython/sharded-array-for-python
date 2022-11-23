// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ddptensor/CppTypes.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>

#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>

namespace jit {

// function type for building body for linalg::generic
using SetResFunc = std::function<void(
    uint64_t rank, void *allocated, void *aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides,
    uint64_t * gs_allocated, uint64_t * gs_aligned, uint64_t * lo_allocated, uint64_t * lo_aligned)>;

// initialize jit
void init();

/// Manages iput/output (tensor) dependences
class DepManager
{
private:
    using IdValueMap = std::unordered_map<id_type, std::pair<::mlir::Value, SetResFunc>>;
    using IdRankMap = std::unordered_map<id_type, int>;
    using ArgList = std::vector<std::pair<id_type, int>>;

    ::mlir::func::FuncOp & _func; // MLIR function to which ops are added
    IdValueMap _ivm;              // guid -> {mlir::Value, deliver-callback}
    IdRankMap _irm;               // guid -> rank as computed in MLIR
    ArgList _args;                // input arguments of the generated function

public:
    DepManager(::mlir::func::FuncOp & f)
    : _func(f)
    {}
    /// @return the ::mlir::Value representing the tensor with guid guid
    /// If the tensor is not created wtihin the current function, it will
    /// be added as a function argument.
    ::mlir::Value getDependent(::mlir::OpBuilder & builder, id_type guid);

    /// for the tensor guid register the ::mlir::value and a callback to deliver the promise which generated the value
    /// if the tensor is alive when the function returns it will be added to the list of results
    void addVal(id_type guid, ::mlir::Value val, SetResFunc cb);

    /// signals end of lifetime of given tensor: does not need to be returned
    void drop(id_type guid);

    /// create return statement and add results to function
    /// this must be called after store_inputs
    /// @return size of output in number of intptr_t's 
    uint64_t handleResult(::mlir::OpBuilder & builder);

    /// devlier promise after execution
    void deliver(intptr_t *, uint64_t);

    /// @return total size of all input arguments in number of intptr_t
    uint64_t arg_size();

    /// store all inputs into given buffer
    /// This must be called before handleResults()
    std::vector<void*> store_inputs();
};

// A class to manage the MLIR business (compilation and execution).
// Just a stub for now, will need to be extended with paramters and maybe more.
class JIT {
public:
    template<typename T, size_t N>
    struct MemRefDescriptor {
        T *allocated = nullptr;
        T *aligned = nullptr;
        intptr_t offset = 0;
        intptr_t sizes[N] = {0};
        intptr_t strides[N] = {0};
    };

    JIT();
    // run
    int run(::mlir::ModuleOp &, const std::string &, std::vector<void*> &, intptr_t *);

    ::mlir::MLIRContext _context;
    ::mlir::PassManager _pm;
    bool _verbose;
};

// size of memreftype in number of intptr_t's
inline uint64_t memref_sz(int rank) { return 3 + 2 * rank; }
inline uint64_t dtensor_sz(int rank) { return 2 * memref_sz(1) + memref_sz(rank) + 1; };

} // namespace jit
