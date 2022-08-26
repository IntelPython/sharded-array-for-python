// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/jit/mlir.hpp"
#include "ddptensor/Registry.hpp"

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

::mlir::Value createI64(const ::mlir::Location & loc, ::mlir::OpBuilder & builder, int64_t val)
{
    auto attr = builder.getI64IntegerAttr(val);
    return builder.create<::mlir::arith::ConstantOp>(loc, attr).getResult();
}

static ::mlir::Type getPTType(::mlir::OpBuilder & builder, DTypeId dtype, int rank)
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

    llvm::SmallVector<int64_t> shape(rank, -1); //::mlir::ShapedType::kDynamicSize);
    return ::imex::ptensor::PTensorType::get(builder.getContext(), ::mlir::RankedTensorType::get(shape, etyp), true);
}

::mlir::Value DepManager::getDependent(::mlir::OpBuilder & builder, id_type guid)
{
    auto loc = builder.getUnknownLoc();
    if(auto d = _ivm.find(guid); d == _ivm.end()) {
        // Not found -> this must be an input argument to the jit function
        auto idx = _args.size();
        auto fut = Registry::get(guid);
        auto typ = getPTType(builder, fut.dtype(), fut.rank());
        _func.insertArgument(idx, typ, {}, loc);
        auto val = _func.getArgument(idx);
        _args.push_back({guid, fut.rank()});
        _ivm[guid] = {val, {}};
        return val;
    } else {
        return d->second.first;
    }
}

// size of memreftype in number of intptr_t's
static inline uint64_t memref_sz(int rank) { return 3 + 2 * rank; }

uint64_t DepManager::arg_size()
{
    uint64_t sz = 0;
    for(auto a : _args) {
        sz += memref_sz(a.second);
    }
    return sz;
}

std::vector<void*> DepManager::store_inputs()
{
    std::vector<void*> res(_args.size());
    int i = 0;
    for(auto a : _args) {
        auto f = Registry::get(a.first);
        intptr_t * buff = new intptr_t[memref_sz(a.second)];
        auto sz = f.get().get()->store_memref(buff, a.second);
        res[i] = buff;
        _ivm.erase(a.first); // inputs need no delivery
        ++i;
    }
    return res;
}

void DepManager::addVal(id_type guid, ::mlir::Value val, SetResFunc cb)
{
    assert(_ivm.find(guid) == _ivm.end());
    _ivm[guid] = {val, cb};
}

void DepManager::drop(id_type guid)
{
    if(auto e = _ivm.find(guid); e != _ivm.end()) {
        _ivm.erase(e);
        // FIXME create delete op
    }
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
    _func.eraseResult(0);

    // here we store the total size of the llvm struct
    uint64_t sz = 0;
    unsigned idx = 0;
    for(auto & v : _ivm) {
        auto value = v.second.first;
        // append the type and array/value
        ret_values[idx] = value;
        _func.insertResult(idx, value.getType(), {});
        auto ptt = value.getType().dyn_cast<::imex::ptensor::PTensorType>();
        assert(ptt);
        auto rank = ptt.getRtensor().getShape().size();
        _irm[v.first] = rank;
        // add sizeof(MemRefDescriptor<elementtype, rank>) to sz
        sz += memref_sz(rank);
        ++idx;
    }

    // add return statement
    auto ret_value = builder.create<::mlir::func::ReturnOp>(builder.getUnknownLoc(), ret_values);

    return sz;
}

void DepManager::deliver(intptr_t * output, uint64_t sz)
{
    size_t pos = 0;
    for(auto & v : _ivm) {
        auto value = v.second.first;
        auto rank = _irm[v.first];
        void * allocated = reinterpret_cast<void*>(output[pos]);
        void * aligned = reinterpret_cast<void*>(output[pos+1]);
        intptr_t offset = output[pos+2];
        intptr_t * sizes = output + pos + 3;
        intptr_t * stride = output + pos + 3 + rank;
        pos += memref_sz(rank);
        v.second.second(rank, allocated, aligned, offset, sizes, stride);
    }
}

int JIT::run(::mlir::ModuleOp & module, const std::string & fname, std::vector<void*> & inp, intptr_t * out)
{    
    if (::mlir::failed(_pm.run(module)))
        throw std::runtime_error("failed to run pass manager");

    module.dump();
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

    llvm::SmallVector<void *> args;
    // first arg must be the result ptr
    args.push_back(&out);
    // we need a void*& for every input tensor
    // we refer directly to the storage in inp
    for(auto & arg : inp) {
        args.push_back(&arg);
    }

    // Invoke the JIT-compiled function.
    if(engine->invokePacked(std::string("_mlir_ciface_") + fn, args)) {
        ::llvm::errs() << "JIT invocation failed\n";
        throw std::runtime_error("JIT invocation failed");
    }

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
    // _pm.enableStatistics();
    // _pm.enableIRPrinting();
    _pm.dump();
}

void init()
{
    assert(sizeof(intptr_t) == sizeof(void*));
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
} // namespace jit
