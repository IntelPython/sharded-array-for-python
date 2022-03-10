import sys, os
sys.path.append(os.path.abspath(os.path.dirname(sys.argv[1])))
exec(f"import {os.path.basename(sys.argv[1]).split('.')[0]} as api")

sys.stdout = open(sys.argv[2], 'w')

print("""// Auto-generated file
// #######################################################
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !! DO NOT EDIT, USE ../scripts/code_gen.py TO UPDATE !!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// #######################################################
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
""")

prev = 0
for cat, lst in api.api_categories.items():
    print(f"enum {cat}Id : int {{")
    for x in lst:
        x = x + f" = {prev}" if x == lst[0] else x
        print(f"    {x.upper()},")
    prev = f"{cat.upper()}_LAST"
    print(f"    {prev}")
    print("};\n")

print("static void def_enums(py::module_ & m)\n{")
for cat, lst in api.api_categories.items():
    print(f'    py::enum_<{cat}Id>(m, "{cat}Id")')
    for x in lst:
        print(f'        .value("{x.upper()}", {x.upper()})')
    print("        .export_values();\n")

print("}")

# Close the file
sys.stdout.close()
