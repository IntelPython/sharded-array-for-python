// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"

class Transceiver
{
public:
    virtual ~Transceiver() {};

    virtual rank_type nranks() const = 0;
    virtual rank_type rank() const = 0;

    // Broadcast data from root to all other processes
    // @param[inout] ptr   on root: pointer to data to be sent
    //                     on all other processes: pointer to buffer to store received data
    // @param[in]    N     number of bytes in ptr
    // @param[in]    root  process id which collects data
    virtual void bcast(void * ptr, size_t N, rank_type root) = 0;

    // Element-wise reduce given array with given operation and provide result on all processes
    // @param[inout] inout input to reduction and result
    // @param[in]    T     data type of elements in inout
    // @param[in]    N     number of elements in inout
    // @param[in]    op    reduction operation
    virtual void reduce_all(void * inout, DType T, size_t N, RedOpType op) = 0;
};

extern Transceiver * theTransceiver;
