// SPDX-License-Identifier: BSD-3-Clause

/*
    helper functions for building the body of linalg::generic/reduce
*/

#pragma once

#include "p2c_ids.hpp"
#include <functional>
#include <mlir/IR/Builders.h>

namespace SHARPY {

// function type for building body for linalg::generic/reduce
using BodyType = std::function<void(
    mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;

extern BodyType getBodyBuilder(EWBinOpId binOp, ::mlir::Type typ);

} // namespace SHARPY
