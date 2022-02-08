import array_api as api

print("""// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
""")

print("enum CreatorId : int {")
for x in api.creators:
    print(f"    {x.upper()},")
print("    CREATOR_LAST")
print("};\n")

uops = api.ew_unary_methods + api.ew_unary_ops
print("enum EWUnyOpId : int {")
for x in uops:
    x = x + " = CREATOR_LAST" if x == uops[0] else x
    print(f"    {x.upper()},")
print("    EWUNYOP_LAST")
print("};\n")

print("enum IEWBinOpId : int {")
for x in api.ew_binary_methods_inplace:
    x = x + " = EWUNYOP_LAST" if x == api.ew_binary_methods_inplace[0] else x
    print(f"    {x.upper()},")
print("    IEWBINOP_LAST")
print("};\n")

bops = api.ew_binary_methods + api.ew_binary_ops
print("enum EWBinOpId : int {")
for x in bops:
    x = x + " = IEWBINOP_LAST" if x == bops[0] else x
    print(f"    {x.upper()},")
print("    EWBINOP_LAST")
print("};\n")

print("enum ReduceOpId : int {")
for x in api.statisticals:
    x = x + " = EWBINOP_LAST" if x == api.statisticals[0] else x
    print(f"    {x.upper()},")
print("    REDUCEOP_LAST")
print("};\n")

print("static void def_enums(py::module_ & m)\n{")

print('    py::enum_<CreatorId>(m, "CreatorId")')
for x in api.creators:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print('    py::enum_<EWUnyOpId>(m, "EWUnyOpId")')
for x in uops:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print('    py::enum_<IEWBinOpId>(m, "IEWBinOpId")')
for x in api.ew_binary_methods_inplace:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print('    py::enum_<EWBinOpId>(m, "EWBinOpId")')
for x in bops:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print('    py::enum_<ReduceOpId>(m, "ReduceOpId")')
for x in api.statisticals:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print("}")
