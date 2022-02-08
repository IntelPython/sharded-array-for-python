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

print("enum IEWBinOpId : int {")
for x in api.ew_binary_methods_inplace:
    x = x[2:-2] + " = CREATOR_LAST" if x == api.ew_binary_methods_inplace[0] else x[2:-2]
    print(f"    {x.upper()},")
print("    IEWBINOP_LAST")
print("};\n")

print("void def_enums(py::module_ & m)\n{")

print('    py::enum_<CreatorId>(m, "CreatorId")')
for x in api.creators:
    print(f'        .value("{x.upper()}", {x.upper()})')
print("        .export_values();\n")

print('    py::enum_<IEWBinOpId>(m, "IEWBinOpId")')
for x in api.ew_binary_methods_inplace:
    print(f'        .value("{x[2:-2].upper()}", {x[2:-2].upper()})')
print("        .export_values();")

print("}")
