// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ddptensor/UtilsAndTypes.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/IR/Builders.h>

#include <unordered_map>
#include <functional>
#include  <utility>

namespace jit {

// function type for building body for linalg::generic
using SetResFunc = std::function<void(
    uint64_t rank, void *allocated, void *aligned, intptr_t offset, intptr_t * sizes, intptr_t * strides)>;
using IdValueMap = std::unordered_map<id_type, std::pair<::mlir::Value, SetResFunc>>;

// initialize jit
void init();

void ttt();

// create a constant integer with given value
extern ::mlir::Value createI64(const ::mlir::Location & loc, ::mlir::OpBuilder & builder, int64_t val);

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
    int run(::mlir::ModuleOp &, const std::string &, void *);

    ::mlir::MLIRContext _context;
    ::mlir::PassManager _pm;
};

} // namespace jit
