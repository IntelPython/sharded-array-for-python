// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "UtilsAndTypes.hpp"
#include "Deferred.hpp"

struct Factory
{
    using ptr_type = std::unique_ptr<Factory>;

    virtual ~Factory() {};
    virtual Deferred::ptr_type create(Deserializer &) const = 0;
    virtual void serialize(Serializer & ser, const Deferred::ptr_type & ptr) const = 0;
    virtual FactoryId id() const = 0;

    template<FactoryId ID> static void init();
    static const Factory * get(FactoryId id);
    static void put(Factory::ptr_type && factory);
};

template<typename D, FactoryId fid>
struct FactoryImpl : public Factory
{
    Deferred::ptr_type create(Deserializer & ser) const
    {
        auto dfrd = std::make_unique<D>();
        dfrd->serialize(ser);
        return dfrd;
    }

    void serialize(Serializer & ser, const Deferred::ptr_type & ptr) const
    {
        D * dfrd = dynamic_cast<D *>(ptr.get());
        if(!dfrd)
            throw std::runtime_error("Invalid Deferred object: dynamic cast failed");
        dfrd->serialize(ser);
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
