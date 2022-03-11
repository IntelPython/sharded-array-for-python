// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>
#include <thread>
#include <iostream>
#include <unordered_map>
#include <mutex>

#include "ddptensor/UtilsAndTypes.hpp"
#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/NDSlice.hpp"
#include "ddptensor/Factory.hpp"

constexpr static int REQ_TAG = 14711;
constexpr static int PULL_TAG = 14712;
constexpr static int PUSH_TAG = 14713;
constexpr static int DEFER_TAG = 14714;
constexpr static int EXIT_TAG = 14715;
static std::mutex ak_mutex;

void send_to_workers(const Deferred::ptr_type & dfrd, bool self = false);

MPIMediator::MPIMediator()
    : _listener(nullptr)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int sz;
    MPI_Comm_size(comm, &sz);
    if(sz > 1)
        _listener = new std::thread(&MPIMediator::listen, this);
}

MPIMediator::~MPIMediator()
{
    std::cerr << "MPIMediator::~MPIMediator()" << std::endl;
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, sz;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &sz);

    if(is_cw() && rank == 0) to_workers(nullptr);
    MPI_Barrier(comm);
    if(!is_cw() || rank == 0) send_to_workers(nullptr, true);
    if(_listener) {
        _listener->join();
        delete _listener;
        _listener = nullptr;
    }
}

void MPIMediator::pull(rank_type from, id_type guid, const NDSlice & slice, void * rbuff)
{
    MPI_Comm comm = MPI_COMM_WORLD;
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

    auto sz = slice.size() * Registry::get(id)->item_size();
    MPI_Irecv(rbuff, sz, MPI_CHAR, from, PUSH_TAG, comm, &request[1]);
    MPI_Isend(buff.data(), cnt, MPI_CHAR, from, REQ_TAG, comm, &request[0]);
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

void send_to_workers(const Deferred::ptr_type & dfrd, bool self)
{
    int rank, sz;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &sz);

    if(rank && ! self)
        throw(std::runtime_error("to_workers assumes controller on rank 0."));

    Buffer buff;
    buff.reserve(256);
    Serializer ser{buff};

    if(dfrd) {
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

    if(self) {
        MPI_Send(buff.data(), cnt, MPI_CHAR, rank, REQ_TAG, comm);
    } else {
        MPI_Request request[sz];
        for(auto i=0; i<sz; ++i) {
            if(i != rank) {
                MPI_Isend(buff.data(), cnt, MPI_CHAR, i, REQ_TAG, comm, &request[i]);
            } else {
                request[i] = MPI_REQUEST_NULL;
            }
        }
        MPI_Waitall(sz, &request[0], MPI_STATUSES_IGNORE);
    }
}

void MPIMediator::to_workers(const Deferred::ptr_type & dfrd)
{
    send_to_workers(dfrd);
}

void MPIMediator::listen()
{
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if(nranks < 2 ) return;

    constexpr int BSZ = 256;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request request_in = MPI_REQUEST_NULL, request_out = MPI_REQUEST_NULL;
    Buffer rbuff;
    // Issue async recv request
    Buffer buff(BSZ);
    MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG, comm, &request_in);
    do {
        MPI_Status status;
        // Wait for any request
        MPI_Wait(&request_in, &status);
        rank_type requester = status.MPI_SOURCE;
        int cnt;
        MPI_Get_count(&status, MPI_CHAR, &cnt);
        buff.resize(cnt);

        Deserializer ser{buff.begin(), cnt};
        int tag;
        ser.value<sizeof(tag)>(tag);
        
        switch(tag) {
        case DEFER_TAG: {
            FactoryId fctryid;
            ser.value<sizeof(fctryid)>(fctryid);
            Deferred::defer(Factory::get(fctryid)->create(ser), true);
            break;
        }
        case PULL_TAG: {
            uint64_t id;
            ser.value8b(id);
            NDSlice slice;
            ser.object(slice);
            
            // Issue async recv request for next msg
            buff.resize(BSZ);
            MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG, comm, &request_in);
            
            // Now find the array in question and send back its bufferized slice
            tensor_i::ptr_type ptr = Registry::get(id);
            // Wait for previous answer to complete so that we can re-use the buffer
            MPI_Wait(&request_out, MPI_STATUS_IGNORE);
            ptr->bufferize(slice, rbuff);
            if(slice.size() * ptr->item_size() != rbuff.size()) throw(std::runtime_error("Got unexpected buffer size."));
            MPI_Isend(rbuff.data(), rbuff.size(), MPI_CHAR, requester, PUSH_TAG, comm, &request_out);
            break;
        }
        case EXIT_TAG:
            Deferred::defer(nullptr, false);
            return;
        default:
            throw(std::runtime_error("Received unexpected message tag."));
        } // switch
        if(request_in == MPI_REQUEST_NULL) {
            // Issue async recv request for next msg
            buff.resize(BSZ);
            MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, REQ_TAG, comm, &request_in);
        }
    } while(true);
    // MPI_Cancel(&request_in);
    // MPI_Wait(&request_out, MPI_STATUS_IGNORE);
}
