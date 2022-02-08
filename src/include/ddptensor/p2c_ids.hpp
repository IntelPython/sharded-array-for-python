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

enum IEWBinOpId : int {
    IADD = CREATOR_LAST,
    IAND,
    IFLOORDIV,
    ILSHIFT,
    IMOD,
    IMUL,
    IOR,
    IPOW,
    IRSHIFT,
    ISUB,
    ITRUEDIV,
    IXOR,
    IEWBINOP_LAST
};

enum EWBinOpId : int {
    ADD = IEWBINOP_LAST,
    AND,
    EQ,
    FLOORDIV,
    GE,
    GT,
    LE,
    LSHIFT,
    LT,
    MATMUL,
    MOD,
    MUL,
    NE,
    OR,
    POW,
    RSHIFT,
    SUB,
    TRUEDIV,
    XOR,
    RADD,
    RAND,
    RFLOORDIV,
    RLSHIFT,
    RMOD,
    RMUL,
    ROR,
    RPOW,
    RRSHIFT,
    RSUB,
    RTRUEDIV,
    RXOR,
    EWBINOP_LAST
};

void def_enums(py::module_ & m)
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

    py::enum_<IEWBinOpId>(m, "IEWBinOpId")
        .value("IADD", IADD)
        .value("IAND", IAND)
        .value("IFLOORDIV", IFLOORDIV)
        .value("ILSHIFT", ILSHIFT)
        .value("IMOD", IMOD)
        .value("IMUL", IMUL)
        .value("IOR", IOR)
        .value("IPOW", IPOW)
        .value("IRSHIFT", IRSHIFT)
        .value("ISUB", ISUB)
        .value("ITRUEDIV", ITRUEDIV)
        .value("IXOR", IXOR)
        .export_values();

    py::enum_<EWBinOpId>(m, "EWBinOpId")
        .value("ADD", ADD)
        .value("AND", AND)
        .value("EQ", EQ)
        .value("FLOORDIV", FLOORDIV)
        .value("GE", GE)
        .value("GT", GT)
        .value("LE", LE)
        .value("LSHIFT", LSHIFT)
        .value("LT", LT)
        .value("MATMUL", MATMUL)
        .value("MOD", MOD)
        .value("MUL", MUL)
        .value("NE", NE)
        .value("OR", OR)
        .value("POW", POW)
        .value("RSHIFT", RSHIFT)
        .value("SUB", SUB)
        .value("TRUEDIV", TRUEDIV)
        .value("XOR", XOR)
        .value("RADD", RADD)
        .value("RAND", RAND)
        .value("RFLOORDIV", RFLOORDIV)
        .value("RLSHIFT", RLSHIFT)
        .value("RMOD", RMOD)
        .value("RMUL", RMUL)
        .value("ROR", ROR)
        .value("RPOW", RPOW)
        .value("RRSHIFT", RRSHIFT)
        .value("RSUB", RSUB)
        .value("RTRUEDIV", RTRUEDIV)
        .value("RXOR", RXOR)
        .export_values();

}
