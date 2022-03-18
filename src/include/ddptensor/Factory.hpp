// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "Deferred.hpp"

struct Factory
{
    using ptr_type = std::unique_ptr<Factory>;

    virtual ~Factory() {};
    virtual Runable::ptr_type create(Deserializer &) const = 0;
    virtual void serialize(Serializer & ser, const Runable * ptr) const = 0;
    virtual FactoryId id() const = 0;

    template<FactoryId ID> static void init();
    static const Factory * get(FactoryId id);
    static void put(Factory::ptr_type && factory);
};

template<typename D, FactoryId fid>
struct FactoryImpl : public Factory
{
    Runable::ptr_type create(Deserializer & ser) const
    {
        auto dfrd = std::make_unique<D>();
        dfrd->serialize(ser);
        return dfrd;
    }

    void serialize(Serializer & ser, const Runable * ptr) const
    {
        auto dfrd = dynamic_cast<const D *>(ptr);
        if(!dfrd)
            throw std::runtime_error("Invalid Deferred object: dynamic cast failed");
        const_cast<D *>(dfrd)->serialize(ser);
    }

    FactoryId id() const
    {
        return fid;
    }

    static void put()
    {
        Factory::put(std::move(std::make_unique<FactoryImpl<D, fid>>()));
    }
};

#define FACTORY_INIT(_D, _ID) \
    template<> void Factory::init<_ID>() { FactoryImpl<_D, _ID>::put(); }
