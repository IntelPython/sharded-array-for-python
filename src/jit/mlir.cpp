// SPDX-License-Identifier: BSD-3-Clause

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

//#include "llvm/ADT/StringRef.h"
//#include "llvm/IR/Module.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/ErrorOr.h"
//#include "llvm/Support/MemoryBuffer.h"
//#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
//#include "llvm/Support/raw_ostream.h"


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

int processMLIR(::mlir::ModuleOp &module)
{
    const char * pl = getenv("DDPT_PASSES");
    // "convert-ptensor-to-linalg,dist-elim,convert-shape-to-std,arith-bufferize,func.func(linalg-init-tensor-to-alloc-tensor,scf-bufferize,shape-bufferize,linalg-bufferize,tensor-bufferize),func-bufferize,func.func(finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,func.func(lower-affine),fold-memref-subview-ops,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,convert-dtensor-to-llvm,reconcile-unrealized-casts",
    if(!pl) pl = "convert-ptensor-to-linalg,dist-elim,convert-shape-to-std,arith-bufferize,func.func(linalg-init-tensor-to-alloc-tensor,scf-bufferize,shape-bufferize,linalg-bufferize,tensor-bufferize),func-bufferize,func.func(finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,func.func(lower-affine),fold-memref-subview-ops,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts";
    ::mlir::PassManager pm(module.getContext());
    if(::mlir::failed(::mlir::parsePassPipeline(pl, pm))) return 3;

    pm.enableStatistics();
    pm.enableIRPrinting();
    pm.dump();
    if (::mlir::failed(pm.run(module))) return 4;
    
    return 0;
}

int runJit(::mlir::ModuleOp & module)
{
    // Initialize LLVM targets.
    ::llvm::InitializeNativeTarget();
    ::llvm::InitializeNativeTargetAsmPrinter();
    //::llvm::initializeLLVMPasses();

    // Register the translation from ::mlir to LLVM IR, which must happen before we
    // can JIT-compile.
    ::mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = ::mlir::makeOptimizingTransformer(0, // /*optLevel=*/enableOpt ? 3 : 0,
                                                       /*sizeLevel=*/0,
                                                       /*targetMachine=*/nullptr);

    // Create an ::mlir execution engine. The execution engine eagerly JIT-compiles
    // the module.
    ::mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = ::mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("ttt_"); //, {{}, {}});
    if (invocationResult) {
        ::llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

void ttt()
{
    std::string fname("_mlir_ttt_");

    ::mlir::registerAllPasses();
    ::imex::registerAllPasses();

    // ::mlir::DialectRegistry registry;
    // ::mlir::registerAllDialects(registry);
    // ::imex::registerAllDialects(registry);

    ::mlir::MLIRContext context(::mlir::MLIRContext::Threading::DISABLED);

    context.getOrLoadDialect<::mlir::arith::ArithmeticDialect>();
    // context.getOrLoadDialect<::mlir::tensor::TensorDialect>();
    // context.getOrLoadDialect<::mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<::mlir::func::FuncDialect>();
    // context.getOrLoadDialect<::mlir::shape::ShapeDialect>();
    context.getOrLoadDialect<::imex::ptensor::PTensorDialect>();
    context.getOrLoadDialect<::imex::dist::DistDialect>();

    ::mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<::mlir::ModuleOp>(loc);

    // Create a func prototype
    auto dtype = builder.getI64Type();
    llvm::SmallVector<int64_t> shape(1, -1); //::mlir::ShapedType::kDynamicSize);
    auto artype = ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(shape, dtype), true);
    auto rrtype = ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(), dtype), true);
    auto funcType = builder.getFunctionType({}, rrtype);
    auto function = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);

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

    auto rangea = builder.create<::imex::ptensor::ARangeOp>(loc, artype, c0, c10, c1, true);
    auto rangeb = builder.create<::imex::ptensor::ARangeOp>(loc, artype, c0, c100, c10, true);
    auto added = builder.create<::imex::ptensor::EWBinOp>(loc, artype, builder.getI32IntegerAttr(::imex::ptensor::ADD), rangea, rangeb);
    auto reduced = builder.create<::imex::ptensor::ReductionOp>(loc, rrtype, builder.getI32IntegerAttr(::imex::ptensor::SUM), added);
    auto ret = builder.create<::mlir::func::ReturnOp>(loc, reduced.getResult());

    module.push_back(function);
    module.dump();

    if(processMLIR(module)) throw std::runtime_error("failed to process mlir");
    module.dump();
    
    if(runJit(module)) throw std::runtime_error("failed to run jit");
    
#if 0                                                  
    std::vector<int> shape = {16, 16};
    auto elemType = builder.getF64Type();
    auto signlessElemType = makeSignlessType(elemType);
    auto indexType = builder.getIndexType();
    auto count = shape.size();
    ::llvm::SmallVector<::mlir::Value> shapeVal(count);
    ::llvm::SmallVector<int64_t> staticShape(count); // ::mlir::ShapedType::kDynamicSize);

    for(auto it : ::llvm::enumerate(shape)) {
        auto i = it.index();
        auto elem = it.value();
        auto elemVal = getInt(loc, builder, elem);
        staticShape[i] = elem;
        shapeVal[i] = elemVal;
    }

    ::mlir::Value init;
    if(true) { //initVal.is_none()) {
        init = builder.create<::mlir::linalg::InitTensorOp>(loc, shapeVal, signlessElemType);
    }//  else {
    //     auto val = doCast(builder, loc, ctx.context.unwrapVal(loc, builder, initVal), signlessElemType);
    //     ::llvm::SmallVector<int64_t> shape(count, ::mlir::ShapedType::kDynamicSize);
    //     auto type = ::mlir::RankedTensorType::get(shape, signlessElemType);
    //     auto body = [&](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange /*indices*/) {
    //         builder.create<::mlir::tensor::YieldOp>(loc, val);
    //     };
    //     init = builder.create<::mlir::tensor::GenerateOp>(loc, type, shapeVal, body);
    // }
    if (::llvm::any_of(staticShape, [](auto val) { return val >= 0; })) {
        auto newType = ::mlir::RankedTensorType::get(staticShape, signlessElemType);
        init = builder.create<::mlir::tensor::CastOp>(loc, newType, init);
    }
    auto resTensorTypeSigness = init.getType().cast<::mlir::RankedTensorType>();
    auto resTensorType = ::mlir::RankedTensorType::get(resTensorTypeSigness.getShape(), elemType, resTensorTypeSigness.getEncoding());
#endif // 0
}
