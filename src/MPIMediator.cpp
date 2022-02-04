// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>
#include <thread>
#include <iostream>
#include <unordered_map>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>

#include "ddptensor/MPIMediator.hpp"
#include "ddptensor/NDSlice.hpp"


using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;
using InputAdapter = bitsery::InputBufferAdapter<Buffer>;
using array_keeper_type = std::unordered_map<uint64_t, tensor_i::ptr_type>;

static array_keeper_type s_ak;
static uint64_t s_last_id = 0;
constexpr static int PULL_TAG = 4711;
constexpr static int PUSH_TAG = 4712;


MPIMediator::MPIMediator()
    : _listener(&MPIMediator::listen, this)
{
}

MPIMediator::~MPIMediator()
{
    std::cerr << "MPIMediator::~MPIMediator()" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Buffer buff;
    bitsery::Serializer<OutputAdapter> ser{buff};
    uint64_t id = 0;
    ser.value8b(id);
    ser.adapter().flush();
    MPI_Send(buff.data(), buff.size(), MPI_CHAR, rank, PULL_TAG, MPI_COMM_WORLD);
    _listener.join();
    s_ak.clear();
}

uint64_t MPIMediator::register_array(tensor_i::ptr_type ary)
{
    s_ak[++s_last_id] = ary;
    return s_last_id;
}

void MPIMediator::pull(rank_type from, const tensor_i * ary, const NDSlice & slice, void * rbuff)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request request[2];
    MPI_Status status[2];
    Buffer buff;

    bitsery::Serializer<OutputAdapter> ser{buff};
    uint64_t id = ary->id();
    ser.value8b(id);
    ser.object(slice);
    ser.adapter().flush();

    auto sz = slice.size() * ary->item_size();
    std::cerr << "alsdkjf " << sz << " " << buff.size() << " " << rbuff << std::endl;
    MPI_Irecv(rbuff, sz, MPI_CHAR, from, PUSH_TAG, comm, &request[1]);
    MPI_Isend(buff.data(), buff.size(), MPI_CHAR, from, PULL_TAG, comm, &request[0]);
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
    int cnt;
    MPI_Get_count(&status[1], MPI_CHAR, &cnt);
    if(cnt != sz) throw(std::runtime_error("Received unexpected message size."));
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
    MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, PULL_TAG, comm, &request_in);
    do {
        MPI_Status status;
        // Wait for any request
        MPI_Wait(&request_in, &status);
        rank_type requester = status.MPI_SOURCE;
        int cnt;
        MPI_Get_count(&status, MPI_CHAR, &cnt);
        buff.resize(cnt);
        bitsery::Deserializer<InputAdapter> ser{buff.begin(), cnt};
        uint64_t id;
        ser.value8b(id);
        if(!id) break;
        NDSlice slice;
        ser.object(slice);

        // Issue async recv request for next msg
        buff.resize(BSZ);
        MPI_Irecv(buff.data(), buff.size(), MPI_CHAR, MPI_ANY_SOURCE, PULL_TAG, comm, &request_in);

        // Now find the array in question and send back its bufferized slice
        auto x = s_ak.find(id);
        if(x == s_ak.end()) throw(std::runtime_error("Encountered pull request for unknown tensor."));
        // Wait for previous answer to complete so that we can re-use the buffer
        MPI_Wait(&request_out, MPI_STATUS_IGNORE);
        x->second->bufferize(slice, rbuff);
        if(slice.size() * x->second->item_size() != rbuff.size()) throw(std::runtime_error("Got unexpected buffer size."));
        MPI_Isend(rbuff.data(), rbuff.size(), MPI_CHAR, requester, PUSH_TAG, comm, &request_out);
    } while(true);
    // MPI_Cancel(&request_in);
    // MPI_Wait(&request_out, MPI_STATUS_IGNORE);
}
