// SPDX-License-Identifier: BSD-3-Clause

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

static mlir::Type makeSignlessType(mlir::Type type)
{
    if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
        auto origElemType = shaped.getElementType();
        auto signlessElemType = makeSignlessType(origElemType);
        return shaped.clone(signlessElemType);
    } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
        if (!intType.isSignless())
            return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
    return type;
}

auto getInt(const mlir::Location & loc, mlir::OpBuilder & builder, int64_t val)
{
    auto attr = builder.getI64IntegerAttr(val);
    return builder.create<mlir::arith::ConstantOp>(loc, attr);
    // auto intType = builder.getIntegerType(64, true);
    // return builder.create<plier::SignCastOp>(loc, intType, res);
}

void ttt()
{
    std::vector<int> shape = {16, 16};
    std::string fname("ttt_mlir");

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    mlir::OpBuilder builder(&context);
    auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto loc = builder.getUnknownLoc();

    // Create a func prototype
    llvm::SmallVector<mlir::Type, 4> argTypes(0);
    auto funcType = builder.getFunctionType(argTypes, llvm::None);
    auto fproto = mlir::FuncOp::create(loc, fname, funcType);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(fproto);
    assert(function);

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);
    
    auto elemType = builder.getF64Type();
    auto signlessElemType = makeSignlessType(elemType);
    auto indexType = builder.getIndexType();
    auto count = shape.size();
    llvm::SmallVector<mlir::Value> shapeVal(count);
    llvm::SmallVector<int64_t> staticShape(count); // mlir::ShapedType::kDynamicSize);

    for(auto it : llvm::enumerate(shape)) {
        auto i = it.index();
        auto elem = it.value();
        auto elemVal = getInt(loc, builder, elem);
        staticShape[i] = elem;
        shapeVal[i] = elemVal;
    }

    mlir::Value init;
    if(true) { //initVal.is_none()) {
        init = builder.create<mlir::linalg::InitTensorOp>(loc, shapeVal, signlessElemType);
    }//  else {
    //     auto val = doCast(builder, loc, ctx.context.unwrapVal(loc, builder, initVal), signlessElemType);
    //     llvm::SmallVector<int64_t> shape(count, mlir::ShapedType::kDynamicSize);
    //     auto type = mlir::RankedTensorType::get(shape, signlessElemType);
    //     auto body = [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange /*indices*/) {
    //         builder.create<mlir::tensor::YieldOp>(loc, val);
    //     };
    //     init = builder.create<mlir::tensor::GenerateOp>(loc, type, shapeVal, body);
    // }
    if (llvm::any_of(staticShape, [](auto val) { return val >= 0; })) {
        auto newType = mlir::RankedTensorType::get(staticShape, signlessElemType);
        init = builder.create<mlir::tensor::CastOp>(loc, newType, init);
    }
    auto resTensorTypeSigness = init.getType().cast<mlir::RankedTensorType>();
    auto resTensorType = mlir::RankedTensorType::get(resTensorTypeSigness.getShape(), elemType, resTensorTypeSigness.getEncoding());
}
