// SPDX-License-Identifier: BSD-3-Clause

#pragma once

class ddptensor;

struct Service
{
    /// replicate the given ddptensor on all ranks
    static ddptensor * replicate(const ddptensor & a);
    /// start running/executing operations, e.g. trigger compile&run
    /// this is not blocking, use futures for synchronization
    static void run();
    /// signal that the given ddptensor is no longer needed and can be deleted
    static void drop(const ddptensor & a);
};
