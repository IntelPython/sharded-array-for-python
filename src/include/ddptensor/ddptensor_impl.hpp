// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cassert>
#include <iostream>
#include <stdlib.h>     /* getenv */
#include <algorithm>

#include "tensor_i.hpp"
#include "PVSlice.hpp"
#include "Transceiver.hpp"
#include "Mediator.hpp"


static tensor_i::ptr_type create_dtensor(const PVSlice & v,
                                         const shape_type & shp,
                                         DType dt,
                                         const char * create = nullptr,
                                         const char * mod = nullptr,
                                         py::args = py::args(),
                                         const py::kwargs & kwargs = py::kwargs());
static tensor_i::ptr_type create_dtensor(const PVSlice & v,
                                         const shape_type & shp,
                                         const char * create,
                                         const char * mod = nullptr,
                                         py::args = py::args(),
                                         const py::kwargs & kwargs = py::kwargs());
static tensor_i::ptr_type create_dtensor(const PVSlice & v,
                                         const shape_type & shp,
                                         py::object pyary,
                                         rank_type owner = NOOWNER);

template<typename T>
tensor_i::ptr_type mktensor(const PVSlice & v,
                            const shape_type & shp,
                            py::object & ary,
                            rank_type owner = NOOWNER);
template<typename T>
tensor_i::ptr_type mktensor(const PVSlice & v,
                            const shape_type & shp,
                            const char * create,
                            const char * mod,
                            py::args args,
                            const py::kwargs & kwargs);



/// The actual distributed tensor implementation.
/// A template class adding types to tensor_i.
/// We can use any array-API compliant python package for node-local computation.
/// This includes allocation which is provided by our dds_allocator.
/// We hold a local python array as well as a DDS NDArray. The latter uses the memory of the first.
/// FIXME: rank i owns tile i only
template<typename T>
class dtensor_impl : public tensor_i
{
    enum : rank_type { NOOWNER = std::numeric_limits<rank_type>::max() };
    typedef std::shared_ptr<dtensor_impl<T>> typed_ptr_type;

    uint64_t    _id;
    uint64_t    _size;
    shape_type  _shape;
    rank_type   _owner;
    PVSlice        _view;
    py::object  _pyarray; // must go first!
    py::module_ _array_ns;
    // std::shared_ptr<dpdlpack> _dpdlpack;

public:
    // Disallow copying our tensor
    dtensor_impl(dtensor_impl<T> &&) = delete;
    dtensor_impl(const dtensor_impl<T> &) = delete;
    dtensor_impl<T> & operator=(const dtensor_impl<T> &) = delete;

    dtensor_impl(const PVSlice & v, const shape_type & shp, py::object & ary, rank_type owner = NOOWNER)
        : _id(0),
          _size(0),
          _shape(shp),
          _owner(owner),
          _view(v),
          _pyarray(ary)
    {
        for(auto s : _shape) _size += s;
        _array_ns = get_array_impl(_pyarray);
    }

    dtensor_impl(const PVSlice & v, const shape_type & shp, const char * create, const char * mod, py::args args, const py::kwargs & kwargs)
        :  _id(0),
           _size(0),
          _shape(shp),
          _owner(NOOWNER),
          _view(v),
          _pyarray(std::move(create_ltensor(_view, create, mod, args, kwargs)))
    {
        for(auto s : _shape) _size += s;
        _array_ns = get_array_impl(_pyarray);
    }

private:
    PVSlice & pvslice()
    {
        return _view;
    }

    py::object create_ltensor(const PVSlice & view, const char * create = "empty", const char * modstr = nullptr,
                              py::args args = py::args(), const py::kwargs & kwargs = py::kwargs()) const
    {
        auto mod = modstr ? py::module_::import(modstr) : get_array_impl(_pyarray);
        // FIXME not all creators accept DType kwarg
        py::dict kwa;
        kwa["dtype"] = get_impl_dtype<T>();
        for(auto x : kwargs) {
            auto key = x.first.cast<std::string>();
            if(key != "dtype") kwa[key.c_str()] = x.second;
        }
        auto x = _make_tuple(view.tile_shape());
        auto xx = mod.attr(create)(x, *args, **kwa);
        return xx;
    }

public:

    const PVSlice & pvslice() const
    {
        return _view;
    }

    void set_id(uint64_t id)
    {
        _id = id;
    }

    uint64_t id() const
    {
        return _id;
    }
  
    int item_size() const
    {
        return sizeof(T);
    }
            
    const shape_type & shape() const
    {
        return _shape;
    }

    uint64_t size() const
    {
        return _size;
    }

    bool is_replicated() const
    {
        return _owner == REPLICATED;
    }

    bool has_owner() const
    {
        return _owner < _OWNER_END;
    }

    void set_owner(rank_type o)
    {
        _owner = o;
    }

    rank_type owner() const
    {
        return _owner;
    }

    void replicate()
    {
        std::cerr << "is_replicated()=" << is_replicated() << " owner=" << owner() << " shape=" << to_string(shape()) << std::endl;
        if(is_replicated()) return;
        if(has_owner()) {
            if(theTransceiver->rank() != owner()) {
                _pyarray = get_array_impl(_pyarray).attr("empty")(1);
            }
            auto ptr = _pyarray.cast<py::buffer>().request().ptr;
            std::cerr << "bcast(" << ptr << ", " << VPROD(shape()) << "*" << sizeof(T) << ", " << owner() << ")" << std::endl;
            theTransceiver->bcast(ptr, VPROD(shape()) * sizeof(T), owner());
            set_owner(REPLICATED);
        } else {
            throw(std::runtime_error("Replication implemented for single owner only."));
        }
    }

    bool __bool__() const
    {
        const_cast<dtensor_impl<T>*>(this)->replicate();
        return _pyarray.attr("__bool__")().cast<bool>();
    }
    
    double __float__() const
    {
        // std::cerr << "__float__ " << py::str(_pyarray).cast<std::string>() << std::endl;
        const_cast<dtensor_impl<T>*>(this)->replicate();
        // std::cerr << "__float__ " << py::str(_pyarray).cast<std::string>() << std::endl;
        return _pyarray.attr("__float__")().cast<double>();
    }

    int64_t __int__() const
    {
        const_cast<dtensor_impl<T>*>(this)->replicate();
        return _pyarray.attr("__int__")().cast<int64_t>();
    }

    shape_type tile_shape() const
    {
        return pvslice().tile_shape();
    }

    uint64_t tile_size(rank_type rank = 0) const
    {
        return pvslice().slice_of_rank(rank).size();
    }

    DType dtype() const
    {
        return DTYPE<T>::value;
    }

    // since the API works on tensor_i we need to downcast to the actual type
    static dtensor_impl<T> * cast(ptr_type & b)
    {
        // FIXME; use attribute/vfunction + reinterpret_cast for better performance
        auto ptr = dynamic_cast<dtensor_impl<T>*>(b.get());
        // if(ptr == nullptr) throw(std::runtime_error("Incompatible tensor types."));
        return ptr;
    }
    static const dtensor_impl<T> * cast(const ptr_type & b)
    {
        return cast(const_cast<ptr_type &>(b));
    }

    ptr_type _ew_op(const char * op, const char * mod, py::args args, const py::kwargs & kwargs)
    {
        return create_dtensor(pvslice(),
                              shape(),
                              (mod ? py::module_::import(mod) : _array_ns).attr(op)(_pyarray, *args, **kwargs));
    }

    ptr_type _ew_unary_op(const char * op, bool is_method) const
    {
        return create_dtensor(pvslice(),
                              shape(),
                              is_method ? _pyarray.attr(op)() : _array_ns.attr(op)(_pyarray));
    }

    ptr_type _ew_binary_op(const char * op, const py::object & b, bool is_method) const
    {
        if(is_method) return create_dtensor(pvslice(), shape(), _pyarray.attr(op)(b));
        else return create_dtensor(pvslice(), shape(), _array_ns.attr(op)(_pyarray, b));
    }

    ptr_type _ew_binary_op(const char * op, const ptr_type & b, bool is_method) const
    {
        auto ab = cast(b);
        if(ab) {
            return _ew_binary_op(op, ab->_pyarray, is_method);
        }
        else throw(std::runtime_error("Not a compatible tensor type."));
    }

    void _ew_binary_op_inplace(const char * op, const py::object & b)
    {
        _pyarray.attr(op)(b);
    }

    void _ew_binary_op_inplace(const char * op, const ptr_type & b)
    {
        auto ab = cast(b);
        if(ab) _ew_binary_op_inplace(op, ab->_pyarray);
        else throw(std::runtime_error("Not a compatible tensor type."));
    }

    ptr_type _reduce_op(const char * op, const dim_vec_type & dims) const
    {
        auto new_shape = reduce_shape(shape(), dims);
        //        auto p = _parter->reduce(dims, shape(), new_shape, theTransceiver->nranks(), theTransceiver->rank(), need_comm);
        py::dict kwa;
        if(dims.empty()) kwa["axis"] = py::none();
        else kwa["axis"] = _make_tuple(dims);
        auto ary = _array_ns.attr(op)(_pyarray, **kwa);
        
        if(pvslice().need_reduce(dims)) {
            py::buffer x = ary.cast<py::buffer>();
            auto buff = x.request();
            auto ptr = buff.ptr;
            auto pylen = VPROD(buff.shape);
            assert(buff.itemsize == sizeof(T));
            theTransceiver->reduce_all(ptr, DTYPE<T>::value, pylen, red_op(op));
            return create_dtensor(pvslice(), new_shape, ary, REPLICATED);
        }

        return create_dtensor(PVSlice(new_shape, pvslice().split_dim()), new_shape, ary, NOOWNER);
    }

    ptr_type __getitem__(const NDSlice & slice) const
    {
        auto nd = shape().size();
        if(nd != slice.ndims())
            throw std::runtime_error("Index dimensionality must match array dimensionality");

        // create a view into orig array
        PVSlice g_slc_view(pvslice(), slice);
        // get the slice for our local partition
        NDSlice lslice;
        if(owner() == REPLICATED) {
            lslice = g_slc_view.slice();
        } else {
            NDSlice my_slice = g_slc_view.slice_of_rank(theTransceiver->rank());
            PVSlice my_view(pvslice(), theTransceiver->rank());
            lslice = my_view.map_slice(my_slice);
        }
        // create py::tuple from slice
        py::tuple tpl;
        if(lslice.ndims()) {
            tpl = _make_tuple(lslice);
        } else {
            _make_tuple(nd, [](const auto & n){return n;}, [](const auto &, int){return py::slice(std::optional<ssize_t>(0), 0, 1);});
        }

        // finally call __getitem__ on local array and create new dist_tensor from it
        auto ln = slice.size();
        std::cerr << "__getitem__ " << slice << " " << to_string(slice.shape()) << " " << ln << " " << lslice << std::endl;
        auto ps = PVSlice(pvslice(), slice);
        if(ln > 1 ) {
            return create_dtensor(ps, ps.shape(), _pyarray.attr("__getitem__")(tpl), NOOWNER);
        } else {
            return create_dtensor(ps, shape_type(), _pyarray.attr("__getitem__")(tpl), pvslice().owner(slice));
        }
    }

    // copy data from val into (*dest)[slice]
    // this is a non-collective call.
    static void _set_slice(const dtensor_impl<T> * val, const NDSlice & val_slice, dtensor_impl<T> * dest, const NDSlice & dest_slice)
    {
        std::cerr << "_set_slice " << val_slice << " " << dest_slice << std::endl;
        auto nd = dest->shape().size();
        if(dest->owner() == REPLICATED && nd > 0)
            std::cerr << "Warning: __setitem__ on replicated data updates local tile only" << std::endl;
        if(nd != dest_slice.ndims())
            throw std::runtime_error("Index dimensionality must match array dimensionality");
        if(val_slice.size() != dest_slice.size())
            throw std::runtime_error("Input and output slices must be of same size");

        // Use given slice to create a global view into orig array
        PVSlice g_slc_view(dest->pvslice(), dest_slice);
        std::cerr << "g_slice: " << g_slc_view.slice() << std::endl;
        // Create a view into val
        PVSlice needed_val_view(val->pvslice(), val_slice);
        std::cerr << "needed_val_view: " << needed_val_view.slice() << " (was " << val->pvslice().slice() << ")" << std::endl;

        // Get the pointer to the local buffer
        auto ns = get_array_impl(dest->_pyarray);

        // we can now compute which ranks actually hold which piece of the data from val that we need locally
        for(rank_type i=0; i<theTransceiver->nranks(); ++i ) {
            // get local view into val
            PVSlice val_local_view(val->pvslice(), i);
            std::cerr << i << " val_local_view: " << val_local_view.slice() << std::endl;
            NDSlice curr_needed_val_slice = needed_val_view.slice_of_rank(i);
            std::cerr << i << " curr_needed_val_slice: " << curr_needed_val_slice << std::endl;
            NDSlice curr_local_val_slice = val_local_view.map_slice(curr_needed_val_slice);
            std::cerr << i << " curr_local_val_slice: " << curr_local_val_slice << std::endl;
            NDSlice curr_needed_norm_slice = needed_val_view.map_slice(curr_needed_val_slice);
            std::cerr << i << " curr_needed_norm_slice: " << curr_needed_norm_slice << std::endl;
            PVSlice my_curr_needed_view = PVSlice(g_slc_view, curr_needed_norm_slice);
            std::cerr << i << " my_curr_needed_slice: " << my_curr_needed_view.slice() << std::endl;
            NDSlice my_curr_local_slice = my_curr_needed_view.local_slice_of_rank(theTransceiver->rank());
            std::cerr << i << " my_curr_local_slice: " << my_curr_local_slice << std::endl;
            if(curr_needed_norm_slice.size()) {
                py::tuple tpl = _make_tuple(my_curr_local_slice); //my_curr_view.slice());
                if(i == theTransceiver->rank()) {
                    // copy locally
                    auto rhs = val->_pyarray.attr("__getitem__")(_make_tuple(curr_local_val_slice));
                    std::cerr << py::str(rhs).cast<std::string>() << std::endl;
                    dest->_pyarray.attr("__setitem__")(tpl, rhs);
                } else {
                    // pull slice directly into new array
                    auto obj = ns.attr("empty")(_make_tuple(curr_local_val_slice.shape()));
                    auto binfo = obj.cast<py::buffer>().request();
                    theMediator->pull(i, val, curr_local_val_slice, binfo.ptr);
                    dest->_pyarray.attr("__setitem__")(tpl, obj);
                }
            }
        }
    }

    // FIXME We use a generic SPMD/PGAS mechanism to pull elements from remote
    // on all procs simultaneously.  Since __setitem__ is collective we could
    // implement a probaly more efficient mechanism which pushes data and/or using RMA.
    void __setitem__(const NDSlice & slice, const ptr_type & val)
    {
        // Use given slice to create a global view into orig array
        PVSlice g_slc_view(this->pvslice(), slice);
        std::cerr << "g_slice: " << g_slc_view.slice() << std::endl;
        NDSlice my_slice = g_slc_view.slice_of_rank(theTransceiver->rank());
        std::cerr << "my_slice: " << my_slice << std::endl;
        NDSlice my_norm_slice = g_slc_view.map_slice(my_slice);
        std::cerr << "my_norm_slice: " << my_norm_slice << std::endl;

        _set_slice(cast(val), my_norm_slice, this, my_slice);
    }

    void bufferize(const NDSlice & slice, Buffer & buff)
    {
        PVSlice my_local_view = PVSlice(tile_shape());
        PVSlice lview = PVSlice(my_local_view, slice);
        NDSlice lslice = lview.slice();

        py::buffer_info info = _pyarray.cast<py::buffer>().request();
        T * ary = reinterpret_cast<T*>(info.ptr);
        for(auto i = 0; i < lslice.ndims(); ++i) {
            std::cerr << "stride[" << i << "]=" << info.strides[i] << std::endl;
        }
        std::cerr << "lslice=" << lslice << " slice= " << slice << " buffsz=" << buff.size() << " want " << slice.size()*sizeof(T) << std::endl;
        buff.resize(slice.size()*sizeof(T));
        T * out = reinterpret_cast<T*>(buff.data());
        int o = 0;
        for(auto i = lview.begin(); i != lview.end(); ++i, ++o) {
            std::cerr << o << " <- " << *i << std::endl;
            out[o] = ary[*i];
        }
    }

    py::object get_slice(const NDSlice & slice) const
    {
        auto shp = slice.shape();
        auto out = create_dtensor(PVSlice(shp, NOSPLIT), shp, DTYPE<T>::value, "empty");
        _set_slice(this, slice, cast(out), {shp});
        return cast(out)->_pyarray;
    }

    std::string __repr__() const
    {
        return "dtensor(shape=" + to_string(shape(), 'x') + ", n_tiles="
            + std::to_string(theTransceiver->nranks()) + ", tile_size=" + to_string(tile_shape(), 'x')
            + ", dtype=" + std::to_string(dtype()) + ")\n" + py::str(_pyarray).cast<std::string>();
    }

};

template<typename TT>
static tensor_i::ptr_type register_tensor(std::shared_ptr<dtensor_impl<TT>> t)
{
    auto id = theMediator->register_array(t);
    t->set_id(id);
    return t;
}

template<typename TT>
static tensor_i::ptr_type mktensor(const PVSlice & v, const shape_type & shp, py::object & ary, rank_type owner)
{
    return register_tensor(std::make_shared<dtensor_impl<TT>>(v, shp, ary, owner));
}

template<typename TT>
static tensor_i::ptr_type mktensor(const PVSlice & v, const shape_type & shp, const char * create, const char * mod, py::args args, const py::kwargs & kwargs)
{
    return register_tensor(std::make_shared<dtensor_impl<TT>>(v, shp, create, mod, args, kwargs));
}

static tensor_i::ptr_type create_dtensor(const PVSlice & v, const shape_type & shp, const char * create, const char * mod, py::args args, const py::kwargs & kwargs)
{
    // FIXME not all creators default to float64
    DType dt = kwargs.contains("dtype") ? static_cast<DType>(kwargs["dtype"].cast<int>()) : DT_FLOAT64;
    return create_dtensor(v, shp, dt, create, mod, args, kwargs);
}

static tensor_i::ptr_type create_dtensor(const PVSlice & v, const shape_type & shp, DType dt, const char * create, const char * mod, py::args args, const py::kwargs & kwargs)
{
    switch(dt) {
    case DT_FLOAT64:
        return mktensor<double>(v, shp, create, mod, args, kwargs);
    case DT_FLOAT32:
        return mktensor<float>(v, shp, create, mod, args, kwargs);
    case DT_INT64:
        return mktensor<int64_t>(v, shp, create, mod, args, kwargs);
    case DT_INT32:
        return mktensor<int32_t>(v, shp, create, mod, args, kwargs);
    case DT_INT16:
        return mktensor<int16_t>(v, shp, create, mod, args, kwargs);
    case DT_UINT64:
        return mktensor<uint64_t>(v, shp, create, mod, args, kwargs);
    case DT_UINT32:
        return mktensor<uint32_t>(v, shp, create, mod, args, kwargs);
    case DT_UINT16:
        return mktensor<uint16_t>(v, shp, create, mod, args, kwargs);
    case DT_BOOL:
        return mktensor<bool>(v, shp, create, mod, args, kwargs);
    default:
        throw std::runtime_error("unknown dtype");
    }
}

static tensor_i::ptr_type create_dtensor(const PVSlice & v, const shape_type & shp, py::object ary, rank_type owner)
{
    py::module_ mod = get_array_impl(ary);
    py::object dt = ary.attr("dtype");
    if(py::hasattr(dt, "type")) dt = dt.attr("type"); // numpy weirdness
    if(dt.is(get_impl_dtype<double>())) return mktensor<double>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<float>())) return mktensor<float>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<int64_t>())) return mktensor<int64_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<int32_t>())) return mktensor<int32_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<int16_t>())) return mktensor<int16_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<uint64_t>())) return mktensor<int64_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<uint32_t>())) return mktensor<int32_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<uint16_t>())) return mktensor<int16_t>(v, shp, ary, owner);
    else if(dt.is(get_impl_dtype<bool>())) return mktensor<bool>(v, shp, ary, owner);
    throw std::runtime_error("unknown dtype");
}
