#include "ddptensor/SetGetItem.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Mediator.hpp"
#include "ddptensor/Factory.hpp"

namespace x {

    class GetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(const NDSlice & slice, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto nd = a_ptr->shape().size();
            if(nd != slice.ndims())
                throw std::runtime_error("Index dimensionality must match array dimensionality");

            return operatorx<T>::mk_tx(*a_ptr.get(), slice);
        }
    };

    class SetItem
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // copy data from val into (*dest)[slice]
        // this is a non-collective call.
        template<typename T, typename X, typename U>
        static void _set_slice(X && dest, const PVSlice & dest_view, const std::shared_ptr<DPTensorX<U>> & val, const NDSlice & val_slice, id_type val_guid)
        {
            auto nd = dest_view.ndims();
            if(val_slice.size() != dest_view.size())
                throw std::runtime_error("Input and output slices must be of same size");

            // Create a view into val
            PVSlice needed_val_view(val->slice(), val_slice);

            // we can now compute which ranks actually hold which piece of the data from val that we need locally
            for(rank_type i=0; i<theTransceiver->nranks(); ++i ) {
                // get local view into val
                PVSlice val_local_view(val->slice(), i);
                NDSlice curr_needed_val_slice = needed_val_view.slice_of_rank(i);
                NDSlice curr_local_val_slice = val_local_view.map_slice(curr_needed_val_slice);
                NDSlice curr_needed_norm_slice = needed_val_view.map_slice(curr_needed_val_slice);
                PVSlice my_curr_needed_view = PVSlice(dest_view, curr_needed_norm_slice);
                NDSlice my_curr_local_slice = my_curr_needed_view.local_slice_of_rank(theTransceiver->rank());

                if(curr_needed_norm_slice.size()) {
                    py::tuple tpl = _make_tuple(my_curr_local_slice); //my_curr_view.slice());
                    if(i == theTransceiver->rank()) {
                        // copy locally
                        auto to_v   = xt::strided_view(dest/*.xarray()*/, to_xt(my_curr_local_slice));
                        auto from_v = xt::strided_view(val->xarray(), to_xt(curr_local_val_slice));
                        to_v = from_v;
                    } else {
                        // pull slice directly into new array
                        xt::xarray<U> from_a = xt::empty<U>(curr_local_val_slice.shape());
                        from_a.fill(static_cast<U>(4711));
                        theMediator->pull(i, val_guid, curr_local_val_slice, from_a.data());
                        auto to_v = xt::strided_view(dest/*.xarray()*/, to_xt(my_curr_local_slice));
                        to_v = from_a;
                    }
                }
            }
        }

        // FIXME We use a generic SPMD/PGAS mechanism to pull elements from remote
        // on all procs simultaneously.  Since __setitem__ is collective we could
        // implement a probaly more efficient mechanism which pushes data and/or using RMA.
        template<typename A, typename B>
        static ptr_type op(const NDSlice & slice, id_type val_guid, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            // Use given slice to create a global view into orig array
            PVSlice g_slc_view(a_ptr->slice(), slice);
            PVSlice my_rel_slice(g_slc_view, theTransceiver->rank());
            NDSlice my_norm_slice = g_slc_view.map_slice(my_rel_slice.slice_of_rank()); //slice());my_slice);
            
            theTransceiver->barrier();
            _set_slice<A>(a_ptr->xarray(), my_rel_slice, b_ptr, my_norm_slice, val_guid);
            return a_ptr;
        }
    };

    class SPMD
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        // get_slice
        template<typename T>
        static py::object op(const NDSlice & slice, id_type val_guid, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto shp = slice.shape();
            auto sz = VPROD(shp);
            auto res = py::array_t<T>(sz);
            auto ax = xt::adapt(res.mutable_data(), sz, xt::no_ownership(), shp);
            PVSlice slc{shp, NOSPLIT};
            SetItem::_set_slice<T>(ax, slc, a_ptr, slice, val_guid);
            return res;
        }

        // get_local
        template<typename T>
        static py::object op(py::handle & handle, const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            auto slc = a_ptr->slice().local_slice_of_rank();
            auto tshp = a_ptr->slice().tile_shape();
            auto nd = slc.ndims();
             // buffer protocol accepts strides in number of bytes not elements!
            std::vector<uint64_t> strides(nd, sizeof(T));
            uint64_t off = slc.dim(nd-1)._start * sizeof(T); // start offset
            for(int i=nd-2; i>=0; --i) {
                auto slci = slc.dim(i);
                auto tmp = strides[i+1] * tshp[i+1];
                strides[i] = slci._step * tmp;
                off += slci._start * tmp;
            }
            off /= sizeof(T); // we need the offset in number of elements
            strides.back() = slc.dim(nd-1)._step * sizeof(T);
            T * data = a_ptr->xarray().data();
            return py::array(std::move(slc.shape()), std::move(strides), data + off, handle);
        }
    };

} // namespace x

struct DeferredSetItem : public Deferred
{
    id_type _a;
    id_type _b;
    NDSlice _slc;
    
    DeferredSetItem() = default;
    DeferredSetItem(tensor_i::future_type & a, const tensor_i::future_type & b, const std::vector<py::slice> & v)
        : _a(a.id()), _b(b.id()), _slc(v)
    {}
    
    void run()
    {
        const auto a = std::move(Registry::get(_a));
        const auto b = std::move(Registry::get(_b));
        set_value(std::move(TypeDispatch<x::SetItem>(a, b, _slc, _b)));
    }

    FactoryId factory() const
    {
        return F_SETITEM;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template value<sizeof(_b)>(_b);
        ser.template object(_slc);
    }
};

tensor_i::future_type SetItem::__setitem__(tensor_i::future_type & a, const std::vector<py::slice> & v, const tensor_i::future_type & b)
{
    return defer<DeferredSetItem>(a, b, v);
}

struct DeferredGetItem : public Deferred
{
    id_type _a;
    NDSlice _slc;

    DeferredGetItem() = default;
    DeferredGetItem(const tensor_i::future_type & a, const std::vector<py::slice> & v)
        : _a(a.id()), _slc(v)
    {}

    void run()
    {
        const auto a = std::move(Registry::get(_a));
        set_value(std::move(TypeDispatch<x::GetItem>(a, _slc)));
    }

    FactoryId factory() const
    {
        return F_GETITEM;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template object(_slc);
    }
};

tensor_i::future_type GetItem::__getitem__(const tensor_i::future_type & a, const std::vector<py::slice> & v)
 {
    return defer<DeferredGetItem>(a, v);
}

py::object GetItem::get_slice(const tensor_i::future_type & a, const std::vector<py::slice> & v)
{
    const auto aa = std::move(a.get());
    return TypeDispatch<x::SPMD>(aa, NDSlice(v), a.id());
}

py::object GetItem::get_local(const tensor_i::future_type & a, py::handle h)
{
    const auto aa = std::move(a.get());
    return TypeDispatch<x::SPMD>(aa, h);
}

FACTORY_INIT(DeferredGetItem, F_GETITEM);
FACTORY_INIT(DeferredSetItem, F_SETITEM);
