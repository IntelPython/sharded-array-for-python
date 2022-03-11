// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "tensor_i.hpp"

/*
  A registry of global tensors.
  Each tensor has a globally unique id.
  FIXME: We currently exchange tensors, not futures, so the
         controlling layer has to make sure dependences are met.
*/
namespace Registry {
    constexpr static id_type NOGUID = -1;

    id_type get_guid();
    extern void put(id_type id, tensor_i::ptr_type ptr);
    tensor_i::ptr_type get(id_type id);
    void del(id_type id);
    void fini();
};
