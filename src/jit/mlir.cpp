// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/jit/mlir.hpp"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "imex/Dialect/PTensor/IR/PTensorOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Pass/PassRegistry.h"

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>

#include <cstdlib>
#include <iostream>

//#include "llvm/ADT/StringRef.h"
//#include "llvm/IR/Module.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/ErrorOr.h"
//#include "llvm/Support/MemoryBuffer.h"
//#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
//#include "llvm/Support/raw_ostream.h"

namespace jit {

static ::mlir::Type makeSignlessType(::mlir::Type type)
{
    if (auto shaped = type.dyn_cast<::mlir::ShapedType>()) {
        auto origElemType = shaped.getElementType();
        auto signlessElemType = makeSignlessType(origElemType);
        return shaped.clone(signlessElemType);
    } else if (auto intType = type.dyn_cast<::mlir::IntegerType>()) {
        if (!intType.isSignless())
            return ::mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
    return type;
}

auto createI64(const ::mlir::Location & loc, ::mlir::OpBuilder & builder, int64_t val)
{
    auto attr = builder.getI64IntegerAttr(val);
    return builder.create<::mlir::arith::ConstantOp>(loc, attr).getResult();
}

int JIT::run(::mlir::ModuleOp & module, const std::string & fname)
{    
    if (::mlir::failed(_pm.run(module)))
        throw std::runtime_error("failed to run pass manager");

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = ::mlir::makeOptimizingTransformer(/*optLevel=*/0,
                                                         /*sizeLevel=*/0,
                                                         /*targetMachine=*/nullptr);

    // Create an ::mlir execution engine. The execution engine eagerly JIT-compiles
    // the module.
    ::mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = ::mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();


    const char * fn = getenv("DDPT_FN");
    if(!fn) fn = fname.c_str();

    MemRefDescriptor<int64_t, 1> result;
    auto r_ptr = &result;
    // int64_t arg = 7;
    // Invoke the JIT-compiled function.
    if(engine->invoke(fn, ::mlir::ExecutionEngine::result(r_ptr))) {
        ::llvm::errs() << "JIT invocation failed\n";
        throw std::runtime_error("JIT invocation failed");
    }
    std::cout << "aptr=" << result.allocated << " dptr=" << result.aligned << " offset=" << result.offset << std::endl;
    std::cout << ((int64_t*)result.aligned)[result.offset] << std::endl;

    return 0;
}

static const char * pass_pipeline =
   getenv("DDPT_PASSES")
   ? getenv("DDPT_PASSES")
   : "convert-ptensor-to-linalg,dist-elim,convert-shape-to-std,arith-bufferize,func.func(linalg-init-tensor-to-alloc-tensor,scf-bufferize,shape-bufferize,linalg-bufferize,tensor-bufferize),func-bufferize,func.func(finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,func.func(lower-affine),fold-memref-subview-ops,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts";
   
JIT::JIT()
    : _context(::mlir::MLIRContext::Threading::DISABLED),
      _pm(&_context)
{
    // Register the translation from ::mlir to LLVM IR, which must happen before we
    // can JIT-compile.
    ::mlir::registerLLVMDialectTranslation(_context);
    // load the dialects we use
    _context.getOrLoadDialect<::mlir::arith::ArithmeticDialect>();
    _context.getOrLoadDialect<::mlir::func::FuncDialect>();
    _context.getOrLoadDialect<::imex::ptensor::PTensorDialect>();
    _context.getOrLoadDialect<::imex::dist::DistDialect>();
    // create the pass pipeline from string
    if(::mlir::failed(::mlir::parsePassPipeline(pass_pipeline, _pm)))
       throw std::runtime_error("failed to parse pass pipeline");
    // some verbosity
    _pm.enableStatistics();
    _pm.enableIRPrinting();
    _pm.dump();
}

void init()
{
    ::mlir::registerAllPasses();
    ::imex::registerAllPasses();

    // ::mlir::DialectRegistry registry;
    // ::mlir::registerAllDialects(registry);
    // ::imex::registerAllDialects(registry);

    // Initialize LLVM targets.
    ::llvm::InitializeNativeTarget();
    ::llvm::InitializeNativeTargetAsmPrinter();
    //::llvm::initializeLLVMPasses();
}

// mock function for POC testing
// delayed execution will do something like the below:
//  * create module
//  * create a function and define its types (input and return types)
//  * create the function body and return op
//  * add function to module
//  * compile & run the module
void ttt()
{
    JIT jit;

    ::mlir::OpBuilder builder(&jit._context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<::mlir::ModuleOp>(loc);

    // Create a func prototype
    auto dtype = builder.getI64Type();
    llvm::SmallVector<int64_t> shape(1, -1); //::mlir::ShapedType::kDynamicSize);
    auto artype = ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(shape, dtype), true);
    auto rrtype = ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(), dtype), true);
    auto funcType = builder.getFunctionType({}, rrtype);

    std::string fname("tttt");
    auto function = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
    // request generation of c-wrapper function
    function->setAttr(::mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(), ::mlir::UnitAttr::get(&jit._context));

    // Create an ::mlir function for the given prototype.
    //::mlir::func::FuncOp function(fproto);
    //assert(function);

    // Let's start the body of the function now!
    // In ::mlir the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);
    
    // create start, stop and step
    auto c0 = createI64(loc, builder, 0);
    auto c10 = createI64(loc, builder, 10);
    auto c1 = createI64(loc, builder, 1);
    auto c100 = createI64(loc, builder, 100);

    // return np.sum(np.arange(1,10,1)+np.arange(1,100,10)) -> 495
    auto rangea = builder.create<::imex::ptensor::ARangeOp>(loc, artype, c0, c10, c1, true);
    auto rangeb = builder.create<::imex::ptensor::ARangeOp>(loc, artype, c0, c100, c10, true);
    auto added = builder.create<::imex::ptensor::EWBinOp>(loc, artype, builder.getI32IntegerAttr(::imex::ptensor::ADD), rangea, rangeb);
    auto reduced = builder.create<::imex::ptensor::ReductionOp>(loc, rrtype, builder.getI32IntegerAttr(::imex::ptensor::SUM), added);
    auto ret = builder.create<::mlir::func::ReturnOp>(loc, reduced.getResult());
    // add the function to the module
    module.push_back(function);

    // finally compile and run the module
    if(jit.run(module, fname)) throw std::runtime_error("failed running jit");
}

} // namespace jit
