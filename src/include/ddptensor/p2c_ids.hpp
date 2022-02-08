// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

enum CreatorId : int {
    ARANGE,
    ASARRAY,
    EMPTY,
    EMPTY_LIKE,
    EYE,
    FROM_DLPACK,
    FULL,
    FULL_LIKE,
    LINSPACE,
    MESHGRID,
    ONES,
    ONES_LIKE,
    ZEROS,
    ZEROS_LIKE,
    CREATOR_LAST
};

enum EWUnyOpId : int {
    __ABS__ = CREATOR_LAST,
    __INVERT__,
    __NEG__,
    __POS__,
    ABS,
    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATANH,
    BITWISE_INVERT,
    CEIL,
    COS,
    COSH,
    EXP,
    EXPM1,
    FLOOR,
    ISFINITE,
    ISINF,
    ISNAN,
    LOGICAL_NOT,
    LOG,
    LOG1P,
    LOG2,
    LOG10,
    NEGATIVE,
    POSITIVE,
    ROUND,
    SIGN,
    SIN,
    SINH,
    SQUARE,
    SQRT,
    TAN,
    TANH,
    TRUNC,
    EWUNYOP_LAST
};

enum IEWBinOpId : int {
    __IADD__ = EWUNYOP_LAST,
    __IAND__,
    __IFLOORDIV__,
    __ILSHIFT__,
    __IMOD__,
    __IMUL__,
    __IOR__,
    __IPOW__,
    __IRSHIFT__,
    __ISUB__,
    __ITRUEDIV__,
    __IXOR__,
    IEWBINOP_LAST
};

enum EWBinOpId : int {
    __ADD__ = IEWBINOP_LAST,
    __AND__,
    __EQ__,
    __FLOORDIV__,
    __GE__,
    __GT__,
    __LE__,
    __LSHIFT__,
    __LT__,
    __MATMUL__,
    __MOD__,
    __MUL__,
    __NE__,
    __OR__,
    __POW__,
    __RSHIFT__,
    __SUB__,
    __TRUEDIV__,
    __XOR__,
    __RADD__,
    __RAND__,
    __RFLOORDIV__,
    __RLSHIFT__,
    __RMOD__,
    __RMUL__,
    __ROR__,
    __RPOW__,
    __RRSHIFT__,
    __RSUB__,
    __RTRUEDIV__,
    __RXOR__,
    ADD,
    ATAN2,
    BITWISE_AND,
    BITWISE_LEFT_SHIFT,
    BITWISE_OR,
    BITWISE_RIGHT_SHIFT,
    BITWISE_XOR,
    DIVIDE,
    EQUAL,
    FLOOR_DIVIDE,
    GREATER,
    GREATER_EQUAL,
    LESS_EQUAL,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    MULTIPLY,
    LESS,
    NOT_EQUAL,
    POW,
    REMAINDER,
    SUBTRACT,
    EWBINOP_LAST
};

enum ReduceOpId : int {
    MAX = EWBINOP_LAST,
    MEAN,
    MIN,
    PROD,
    SUM,
    STD,
    VAR,
    REDUCEOP_LAST
};

static void def_enums(py::module_ & m)
{
    py::enum_<CreatorId>(m, "CreatorId")
        .value("ARANGE", ARANGE)
        .value("ASARRAY", ASARRAY)
        .value("EMPTY", EMPTY)
        .value("EMPTY_LIKE", EMPTY_LIKE)
        .value("EYE", EYE)
        .value("FROM_DLPACK", FROM_DLPACK)
        .value("FULL", FULL)
        .value("FULL_LIKE", FULL_LIKE)
        .value("LINSPACE", LINSPACE)
        .value("MESHGRID", MESHGRID)
        .value("ONES", ONES)
        .value("ONES_LIKE", ONES_LIKE)
        .value("ZEROS", ZEROS)
        .value("ZEROS_LIKE", ZEROS_LIKE)
        .export_values();

    py::enum_<EWUnyOpId>(m, "EWUnyOpId")
        .value("__ABS__", __ABS__)
        .value("__INVERT__", __INVERT__)
        .value("__NEG__", __NEG__)
        .value("__POS__", __POS__)
        .value("ABS", ABS)
        .value("ACOS", ACOS)
        .value("ACOSH", ACOSH)
        .value("ASIN", ASIN)
        .value("ASINH", ASINH)
        .value("ATAN", ATAN)
        .value("ATANH", ATANH)
        .value("BITWISE_INVERT", BITWISE_INVERT)
        .value("CEIL", CEIL)
        .value("COS", COS)
        .value("COSH", COSH)
        .value("EXP", EXP)
        .value("EXPM1", EXPM1)
        .value("FLOOR", FLOOR)
        .value("ISFINITE", ISFINITE)
        .value("ISINF", ISINF)
        .value("ISNAN", ISNAN)
        .value("LOGICAL_NOT", LOGICAL_NOT)
        .value("LOG", LOG)
        .value("LOG1P", LOG1P)
        .value("LOG2", LOG2)
        .value("LOG10", LOG10)
        .value("NEGATIVE", NEGATIVE)
        .value("POSITIVE", POSITIVE)
        .value("ROUND", ROUND)
        .value("SIGN", SIGN)
        .value("SIN", SIN)
        .value("SINH", SINH)
        .value("SQUARE", SQUARE)
        .value("SQRT", SQRT)
        .value("TAN", TAN)
        .value("TANH", TANH)
        .value("TRUNC", TRUNC)
        .export_values();

    py::enum_<IEWBinOpId>(m, "IEWBinOpId")
        .value("__IADD__", __IADD__)
        .value("__IAND__", __IAND__)
        .value("__IFLOORDIV__", __IFLOORDIV__)
        .value("__ILSHIFT__", __ILSHIFT__)
        .value("__IMOD__", __IMOD__)
        .value("__IMUL__", __IMUL__)
        .value("__IOR__", __IOR__)
        .value("__IPOW__", __IPOW__)
        .value("__IRSHIFT__", __IRSHIFT__)
        .value("__ISUB__", __ISUB__)
        .value("__ITRUEDIV__", __ITRUEDIV__)
        .value("__IXOR__", __IXOR__)
        .export_values();

    py::enum_<EWBinOpId>(m, "EWBinOpId")
        .value("__ADD__", __ADD__)
        .value("__AND__", __AND__)
        .value("__EQ__", __EQ__)
        .value("__FLOORDIV__", __FLOORDIV__)
        .value("__GE__", __GE__)
        .value("__GT__", __GT__)
        .value("__LE__", __LE__)
        .value("__LSHIFT__", __LSHIFT__)
        .value("__LT__", __LT__)
        .value("__MATMUL__", __MATMUL__)
        .value("__MOD__", __MOD__)
        .value("__MUL__", __MUL__)
        .value("__NE__", __NE__)
        .value("__OR__", __OR__)
        .value("__POW__", __POW__)
        .value("__RSHIFT__", __RSHIFT__)
        .value("__SUB__", __SUB__)
        .value("__TRUEDIV__", __TRUEDIV__)
        .value("__XOR__", __XOR__)
        .value("__RADD__", __RADD__)
        .value("__RAND__", __RAND__)
        .value("__RFLOORDIV__", __RFLOORDIV__)
        .value("__RLSHIFT__", __RLSHIFT__)
        .value("__RMOD__", __RMOD__)
        .value("__RMUL__", __RMUL__)
        .value("__ROR__", __ROR__)
        .value("__RPOW__", __RPOW__)
        .value("__RRSHIFT__", __RRSHIFT__)
        .value("__RSUB__", __RSUB__)
        .value("__RTRUEDIV__", __RTRUEDIV__)
        .value("__RXOR__", __RXOR__)
        .value("ADD", ADD)
        .value("ATAN2", ATAN2)
        .value("BITWISE_AND", BITWISE_AND)
        .value("BITWISE_LEFT_SHIFT", BITWISE_LEFT_SHIFT)
        .value("BITWISE_OR", BITWISE_OR)
        .value("BITWISE_RIGHT_SHIFT", BITWISE_RIGHT_SHIFT)
        .value("BITWISE_XOR", BITWISE_XOR)
        .value("DIVIDE", DIVIDE)
        .value("EQUAL", EQUAL)
        .value("FLOOR_DIVIDE", FLOOR_DIVIDE)
        .value("GREATER", GREATER)
        .value("GREATER_EQUAL", GREATER_EQUAL)
        .value("LESS_EQUAL", LESS_EQUAL)
        .value("LOGADDEXP", LOGADDEXP)
        .value("LOGICAL_AND", LOGICAL_AND)
        .value("LOGICAL_OR", LOGICAL_OR)
        .value("LOGICAL_XOR", LOGICAL_XOR)
        .value("MULTIPLY", MULTIPLY)
        .value("LESS", LESS)
        .value("NOT_EQUAL", NOT_EQUAL)
        .value("POW", POW)
        .value("REMAINDER", REMAINDER)
        .value("SUBTRACT", SUBTRACT)
        .export_values();

    py::enum_<ReduceOpId>(m, "ReduceOpId")
        .value("MAX", MAX)
        .value("MEAN", MEAN)
        .value("MIN", MIN)
        .value("PROD", PROD)
        .value("SUM", SUM)
        .value("STD", STD)
        .value("VAR", VAR)
        .export_values();

}
