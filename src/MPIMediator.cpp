// SPDX-License-Identifier: BSD-3-Clause

/*
  A high-level mediation between processes/ranks implemented on top of MPI.
*/

#include <iostream>
#include <mpi.h>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "sharpy/CppTypes.hpp"
#include "sharpy/Factory.hpp"
#include "sharpy/MPIMediator.hpp"
#include "sharpy/MPITransceiver.hpp"

namespace SHARPY {

constexpr static int REQ_TAG = 14711;
constexpr static int PULL_TAG = 14712;
constexpr static int PUSH_TAG = 14713;
constexpr static int DEFER_TAG = 14714;
constexpr static int EXIT_TAG = 14715;
static std::mutex ak_mutex;

void send_to_workers(const Runable *dfrd, bool self, MPI_Comm comm);

MPIMediator::MPIMediator() : _listener(nullptr) {
  auto c = dynamic_cast<MPITransceiver *>(getTransceiver());
  if (c == nullptr)
    throw std::runtime_error("Expected Transceiver to be MPITransceiver.");
  _comm = c->comm();
  int sz;
  MPI_Comm_size(_comm, &sz);
  if (sz > 1 && getTransceiver()->is_cw())
    _listener = new std::thread(&MPIMediator::listen, this);
}

MPIMediator::~MPIMediator() {
  std::cerr << "MPIMediator::~MPIMediator()" << std::endl;
  int rank, sz;
  MPI_Comm_rank(_comm, &rank);
  MPI_Comm_size(_comm, &sz);

  if (getTransceiver()->is_cw() && rank == 0)
    to_workers(nullptr);
  MPI_Barrier(_comm);
  if (!getTransceiver()->is_cw() || rank == 0)
    defer(nullptr); // send_to_workers(nullptr, true, _comm);
  if (_listener) {
    _listener->join();
    delete _listener;
    _listener = nullptr;
  }
}

#if 0
void MPIMediator::pull(rank_type from, id_type guid, const NDSlice & slice, void * rbuff)
{
    MPI_Request request[2];
    MPI_Status status[2];
    Buffer buff;

    Serializer ser{buff};
    auto id = guid;
    int tag = PULL_TAG;
    ser.value<sizeof(tag)>(tag);
    ser.value8b(id);
    ser.object(slice);
    ser.adapter().flush();
    int cnt = static_cast<int>(ser.adapter().writtenBytesCount());

    auto sz = slice.size() * Registry::get(id).get()->item_size();
    MPI_Irecv(rbuff, sz, MPI_CHAR, from, PUSH_TAG, _comm, &request[1]);
    MPI_Isend(buff.data(), cnt, MPI_CHAR, from, REQ_TAG, _comm, &request[0]);
    auto error_code = MPI_Waitall(2, &request[0], &status[0]);
    if (error_code != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Waitall returned error code " + std::to_string(error_code));
    }
    if(status[0].MPI_ERROR != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Waitall returned error code " + std::to_string(status[0].MPI_ERROR));
    }
    if(status[1].MPI_ERROR != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Waitall returned error code " + std::to_string(status[1].MPI_ERROR));
    }
    MPI_Get_count(&status[1], MPI_CHAR, &cnt);
    if(cnt != sz) throw(std::runtime_error("Received unexpected message size."));
}
#endif

void send_to_workers(const Runable *dfrd, bool self, MPI_Comm comm) {
  int rank, sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &sz);

  if (rank && !self)
    throw(std::runtime_error("to_workers assumes controller on rank 0."));

  Buffer buff;
  buff.reserve(256);
  Serializer ser{buff};

  if (dfrd) {
    const auto fctry = Factory::get(dfrd->factory());
    int tag = DEFER_TAG;
    auto fid = fctry->id();
    ser.value<sizeof(tag)>(tag);
    ser.value<sizeof(fid)>(fid);
    fctry->serialize(ser, dfrd);
  } else {
    int tag = EXIT_TAG;
    ser.value<sizeof(tag)>(tag);
  }
  ser.adapter().flush();
  int cnt = static_cast<int>(ser.adapter().writtenBytesCount());

  if (self) {
    MPI_Send(buff.data(), cnt, MPI_CHAR, rank, REQ_TAG, comm);
  } else {
    MPI_Request request[sz];
    for (auto i = 0; i < sz; ++i) {
      if (i != rank) {
        MPI_Isend(buff.data(), cnt, MPI_CHAR, i, REQ_TAG, comm, &request[i]);
      } else {
        request[i] = MPI_REQUEST_NULL;
      }
    }
    MPI_Waitall(sz, &request[0], MPI_STATUSES_IGNORE);
  }
}

void MPIMediator::to_workers(const Runable *dfrd) {
  send_to_workers(dfrd, false, _comm);
}

void MPIMediator::listen() {
  int nranks;
  MPI_Comm_size(_comm, &nranks);
  if (nranks < 2)
    return;

  constexpr int BSZ = 256;
  MPI_Request request_in = MPI_REQUEST_NULL;
  Buffer rbuff;
  // Issue async recv request
  Buffer buff(BSZ);
  MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG, _comm,
            &request_in);
  do {
    MPI_Status status;
    // Wait for any request
    MPI_Wait(&request_in, &status);
    int cnt;
    MPI_Get_count(&status, MPI_CHAR, &cnt);
    buff.resize(cnt);

    Deserializer ser{buff.begin(), static_cast<size_t>(cnt)};
    int tag;
    ser.value<sizeof(tag)>(tag);

    switch (tag) {
    case DEFER_TAG: {
      FactoryId fctryid;
      ser.value<sizeof(fctryid)>(fctryid);
      auto uptr = Factory::get(fctryid)->create(ser);
      uptr.get()->defer(std::move(uptr)); // grmpf
      break;
    }
#if 0
    rank_type requester = status.MPI_SOURCE;
    MPI_Request request_out = MPI_REQUEST_NULL;
        case PULL_TAG: {
            uint64_t id;
            ser.value8b(id);
            NDSlice slice;
            ser.object(slice);

            // Issue async recv request for next msg
            buff.resize(BSZ);
            MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG, _comm, &request_in);

            // Now find the array in question and send back its bufferized slice
            array_i::ptr_type ptr = Registry::get(id).get();
            // Wait for previous answer to complete so that we can re-use the buffer
            MPI_Wait(&request_out, MPI_STATUS_IGNORE);
            rbuff.resize(0);
            ptr->bufferize(slice, rbuff);
            if(slice.size() * ptr->item_size() != rbuff.size()) throw(std::runtime_error("Got unexpected buffer size."));
            MPI_Isend(rbuff.data(), rbuff.size(), MPI_CHAR, requester, PUSH_TAG, _comm, &request_out);
            break;
        }
#endif
    case EXIT_TAG:
      defer(nullptr);
      return;
    default:
      throw(std::runtime_error("Received unexpected message tag."));
    } // switch
    if (request_in == MPI_REQUEST_NULL) {
      // Issue async recv request for next msg
      buff.resize(BSZ);
      MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG,
                _comm, &request_in);
    }
  } while (true);
  // MPI_Cancel(&request_in);
  // MPI_Wait(&request_out, MPI_STATUS_IGNORE);
}
} // namespace SHARPY
