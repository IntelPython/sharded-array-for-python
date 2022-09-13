
// SPDX-License-Identifier: BSD-3-Clause

#include <ddptensor/Transceiver.hpp>

Transceiver * theTransceiver = nullptr;

void init_transceiver(Transceiver * t)
 {
    if(theTransceiver) delete theTransceiver;
    theTransceiver = t;
}

void fini_transceiver()
{
    if(theTransceiver) delete theTransceiver;
    theTransceiver = nullptr;
}

Transceiver * getTransceiver()
{
    return theTransceiver;
}
