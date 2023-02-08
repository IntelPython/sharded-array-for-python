// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/jit/mlir.hpp"
#include "ddptensor/Registry.hpp"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
// #include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
// #include "mlir/Dialect/GPU/Transforms/Passes.h"
// #include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
// #include "mlir/Dialect/NVGPU/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
// #include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
// #include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
// #include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
// #include "mlir/Dialect/Tosa/Transforms/Passes.h"
// #include "mlir/Dialect/Transform/Transforms/Passes.h"
// #include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
// #include <mlir/InitAllPasses.h>


#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
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

static ::mlir::Type getDTType(::mlir::OpBuilder & builder, DTypeId dtype, int rank, bool balanced)
{
    ::mlir::Type etyp;

    switch(dtype) {
    case FLOAT64:
        etyp = builder.getF64Type();
        break;
    case FLOAT32:
        etyp = builder.getF32Type();
        break;
    case INT64:
    case UINT64:
        etyp = builder.getI64Type();
        break;
    case INT32:
    case UINT32:
        etyp = builder.getI32Type();
        break;
    case INT16:
    case UINT16:
        etyp = builder.getIntegerType(16);
        break;
    case INT8:
    case UINT8:
        etyp = builder.getI8Type();
        break;
    case BOOL:
        etyp = builder.getI1Type();
        break;
    default:
        throw std::runtime_error("unknown dtype");
    };

    return ::imex::dist::DistTensorType::get(
        builder.getContext(),
        ::imex::ptensor::PTensorType::get(builder.getContext(), rank, etyp, false)
    );
}

::mlir::Value DepManager::getDependent(::mlir::OpBuilder & builder, id_type guid)
{
    auto loc = builder.getUnknownLoc();
    if(auto d = _ivm.find(guid); d == _ivm.end()) {
        // Not found -> this must be an input argument to the jit function
        auto idx = _args.size();
        auto fut = Registry::get(guid);
        auto typ = getDTType(builder, fut.dtype(), fut.rank(), fut.balanced());
        _func.insertArgument(idx, typ, {}, loc);
        auto val = _func.getArgument(idx);
        _args.push_back({guid, fut.rank()});
        _ivm[guid] = val;
        return val;
    } else {
        return d->second;
    }
}

uint64_t DepManager::arg_size()
{
    uint64_t sz = 0;
    for(auto a : _args) {
        sz += dtensor_sz(a.second);
    }
    return sz;
}

std::vector<void*> DepManager::store_inputs()
{
    std::vector<void*> res;
    for(auto a : _args) {
        auto f = Registry::get(a.first);
        f.get().get()->add_to_args(res, a.second);
        _ivm.erase(a.first); // inputs need no delivery
        _icm.erase(a.first);
    }
    return res;
}

void DepManager::addVal(id_type guid, ::mlir::Value val, SetResFunc cb)
{
    assert(_ivm.find(guid) == _ivm.end());
    _ivm[guid] = val;
    _icm[guid] = cb;
}

void DepManager::drop(id_type guid)
{
    _ivm.erase(guid);
    _icm.erase(guid);
    // FIXME create delete op
}

// Now we have to define the return type as a ValueRange of all arrays which we have created
// (runnables have put them into DepManager when generating mlir)
// We also compute the total size of the struct llvm created for this return type
// llvm will basically return a struct with all the arrays as members, each of type JIT::MemRefDescriptor
uint64_t DepManager::handleResult(::mlir::OpBuilder & builder)
{
    // Need a container to put all return values, will be used to construct TypeRange
    std::vector<::mlir::Value> ret_values(_ivm.size());

    // remove default result
    //_func.eraseResult(0);

    // here we store the total size of the llvm struct
    auto loc = builder.getUnknownLoc();
    uint64_t sz = 0;
    unsigned idx = 0;
    for(auto & v : _ivm) {
        ::mlir::Value value = v.second;
        // append the type and array/value
        auto retDtTyp = value.getType().dyn_cast<::imex::dist::DistTensorType>();
        if(!retDtTyp) {
            auto valPtTyp = value.getType().dyn_cast<::imex::ptensor::PTensorType>();
            assert(valPtTyp);
            retDtTyp = ::imex::dist::DistTensorType::get(builder.getContext(), valPtTyp); // FIXME balanced needs to be reported back
            value = builder.create<::mlir::UnrealizedConversionCastOp>(loc, retDtTyp, value).getResult(0);
        }
        ret_values[idx] = value;
        _func.insertResult(idx, value.getType(), {});
        auto rank = retDtTyp.getPTensorType().getRank();
        _irm[v.first] = rank;
        // add sizes of dtensor (3 memrefs + team + balanced) to sz
        sz += dtensor_sz(rank);
        ++idx;
    }

    // add return statement
    auto ret_value = builder.create<::mlir::func::ReturnOp>(builder.getUnknownLoc(), ret_values);

    // clear any reference to MLIR values
    _ivm.clear();
    return 2*sz;
}

void DepManager::deliver(intptr_t * output, uint64_t sz)
{
    size_t pos = 0;
    for(auto & v : _icm) {
        auto rank = _irm[v.first];
        // first extract tensor
        void * t_allocated = reinterpret_cast<void*>(output[pos]);
        void * t_aligned = reinterpret_cast<void*>(output[pos+1]);
        intptr_t t_offset = output[pos+2];
        intptr_t * t_sizes = output + pos + 3;
        intptr_t * t_stride = output + pos + 3 + rank;
        pos += memref_sz(rank);
        // second is the team
        auto team = output[pos];
        pos += 1;
        // third is balanced
        auto balanced = output[pos];
        pos += 1;
        if(rank > 0) {
            // third extract global shape
            uint64_t * gs_allocated = reinterpret_cast<uint64_t*>(output[pos]);
            uint64_t * gs_aligned = reinterpret_cast<uint64_t*>(output[pos+1]);
            intptr_t gs_offset = output[pos+2];
            // no sizes/stride needed
            pos += memref_sz(1);
            // lastly extract local offsets
            uint64_t * lo_allocated = reinterpret_cast<uint64_t*>(output[pos]);
            uint64_t * lo_aligned = reinterpret_cast<uint64_t*>(output[pos+1]);
            intptr_t lo_offset = output[pos+2];
            // no sizes/stride needed
            pos += memref_sz(1);
            // call finalization
            v.second(reinterpret_cast<Transceiver*>(team), rank,
                     t_allocated, t_aligned, t_offset, t_sizes, t_stride, // tensor
                     gs_allocated, gs_aligned + gs_offset, // global shape is 1d tensor of uint64_t
                     lo_allocated, lo_aligned + lo_offset, // local offset is 1d tensor of uint64_t
                     balanced
            );
        } else { // 0d tensor
            v.second(reinterpret_cast<Transceiver*>(team), rank, t_allocated, t_aligned, t_offset, t_sizes, t_stride,
                     nullptr, nullptr, nullptr, nullptr, 1);
        }
    }
}

int JIT::run(::mlir::ModuleOp & module, const std::string & fname, std::vector<void*> & inp, intptr_t * out)
{
    // lower to LLVM
    if (::mlir::failed(_pm.run(module)))
        throw std::runtime_error("failed to run pass manager");

    if(_verbose) module.dump();

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = ::mlir::makeOptimizingTransformer(/*optLevel=*/0,
                                                         /*sizeLevel=*/0,
                                                         /*targetMachine=*/nullptr);

    // Create an ::mlir execution engine. The execution engine eagerly JIT-compiles
    // the module.
    ::mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    // const char * crunner = getenv("DDPT_CRUNNER_SO");
    // crunner = crunner ? crunner : "libmlir_c_runner_utils.so";
    const char * idtr = getenv("DDPT_IDTR_SO");
    idtr = idtr ? idtr : "libidtr.so";
    // ::llvm::ArrayRef<::llvm::StringRef> shlibs = {crunner, idtr};
    engineOptions.sharedLibPaths = {idtr};
    auto maybeEngine = ::mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    llvm::SmallVector<void *> args;
    // first arg must be the result ptr
    args.push_back(&out);
    // we need a void*& for every input tensor
    // we refer directly to the storage in inp
    for(auto & arg : inp) {
        args.push_back(&arg);
    }

    // Invoke the JIT-compiled function.
    if(engine->invokePacked(std::string("_mlir_ciface_") + fname.c_str(), args)) {
        ::llvm::errs() << "JIT invocation failed\n";
        throw std::runtime_error("JIT invocation failed");
    }

    return 0;
}

static const char * pass_pipeline =
   getenv("DDPT_PASSES")
   ? getenv("DDPT_PASSES")
//    : "func.func(ptensor-dist),convert-dist-to-standard,convert-ptensor-to-linalg,arith-expand,canonicalize,arith-bufferize,func.func(empty-tensor-to-alloc-tensor,scf-bufferize,linalg-bufferize,tensor-bufferize),func-bufferize,canonicalize,func.func(finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,fold-memref-alias-ops,lower-affine,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts";
//    : "builtin.module(func.func(ptensor-dist),convert-dist-to-standard,convert-ptensor-to-linalg,arith-bufferize,func.func(empty-tensor-to-alloc-tensor,scf-bufferize,linalg-bufferize,tensor-bufferize,bufferization-bufferize),func-bufferize,func.func(finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,fold-memref-alias-ops,expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)";
      : "func.func(ptensor-dist,dist-coalesce),convert-dist-to-standard,convert-ptensor-to-linalg,canonicalize,convert-shape-to-std,arith-expand,canonicalize,arith-bufferize,func-bufferize,func.func(empty-tensor-to-alloc-tensor,scf-bufferize,tensor-bufferize,linalg-bufferize,bufferization-bufferize,linalg-detensorize,tensor-bufferize,finalizing-bufferize,convert-linalg-to-parallel-loops),canonicalize,fold-memref-alias-ops,expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts";
JIT::JIT()
    : _context(::mlir::MLIRContext::Threading::DISABLED),
      _pm(&_context),
      _verbose(false)
{
    // Register the translation from ::mlir to LLVM IR, which must happen before we
    // can JIT-compile.
    ::mlir::registerLLVMDialectTranslation(_context);
    // load the dialects we use
    _context.getOrLoadDialect<::mlir::arith::ArithDialect>();
    _context.getOrLoadDialect<::mlir::func::FuncDialect>();
    _context.getOrLoadDialect<::mlir::linalg::LinalgDialect>();
    _context.getOrLoadDialect<::imex::ptensor::PTensorDialect>();
    _context.getOrLoadDialect<::imex::dist::DistDialect>();
    // create the pass pipeline from string
    if(::mlir::failed(::mlir::parsePassPipeline(pass_pipeline, _pm)))
       throw std::runtime_error("failed to parse pass pipeline");

    const char * v_ = getenv("DDPT_VERBOSE");
    if(v_) {
        std::string v(v_);
        if(v == "1" || v == "y" || v == "Y" || v == "on" || v == "ON") _verbose = true;
    }
    // some verbosity
    if(_verbose) {
        _pm.enableStatistics();
        _pm.enableIRPrinting();
        _pm.dump();
    }
}

void init()
{
    assert(sizeof(intptr_t) == sizeof(void*));
    assert(sizeof(intptr_t) == sizeof(uint64_t));
    // ::mlir::registerAllPasses();
    ::mlir::registerSCFPasses();
    ::mlir::registerSCFToControlFlowPass();
    ::mlir::registerShapePasses();
    ::mlir::registerConvertShapeToStandardPass();
    ::mlir::tensor::registerTensorPasses();
    ::mlir::registerLinalgPasses();
    ::mlir::func::registerFuncPasses();
    ::mlir::registerConvertFuncToLLVMPass();
    ::mlir::bufferization::registerBufferizationPasses();
    ::mlir::arith::registerArithPasses();
    ::mlir::registerAffinePasses();
    ::mlir::registerMemRefToLLVMConversionPass();
    ::mlir::registerCanonicalizerPass();
    ::mlir::registerConvertAffineToStandardPass();
    ::mlir::memref::registerMemRefPasses();
    ::mlir::registerReconcileUnrealizedCastsPass();

    ::imex::registerPTensorPasses();
    ::imex::registerDistPasses();
    ::imex::registerConvertDistToStandard();
    ::imex::registerConvertPTensorToLinalg();

    // ::mlir::DialectRegistry registry;
    // ::mlir::registerAllDialects(registry);
    // ::imex::registerAllDialects(registry);

    // Initialize LLVM targets.
    ::llvm::InitializeNativeTarget();
    ::llvm::InitializeNativeTargetAsmPrinter();
    //::llvm::initializeLLVMPasses();
}
} // namespace jit
