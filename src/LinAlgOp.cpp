#include <mpi.h>
//#include <mkl.h>
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace x {

    template<typename T> struct TGEMM;
    template<> struct TGEMM<double> { static constexpr auto tgemm = cblas_dgemm; };
    template<> struct TGEMM<float> { static constexpr auto tgemm = cblas_sgemm; };

    class LinAlgOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename A, typename B>
        static ptr_type op(int axis, const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            if constexpr (std::is_floating_point<A>::value && std::is_same<A, B>::value) {
                const auto & ax = a_ptr->xarray();
                const auto & bx = b_ptr->xarray();
                auto nda = a_ptr->slice().ndims();
                auto ndb = b_ptr->slice().ndims();
                
                if(a_ptr->is_sliced() || b_ptr->is_sliced()) {
                    if(nda != 1 || ndb != 1)
                        throw(std::runtime_error("vecdoc on sliced tensors supported for 1d tensors only"));
                    const auto & av = xt::strided_view(ax, a_ptr->lslice());
                    const auto & bv = xt::strided_view(bx, b_ptr->lslice());
                    return vecdot_1d(av, bv, axis);
                }
                
                if(nda == 1 && ndb == 1) {
                    return vecdot_1d(ax, bx, axis);
                } else if(nda == 2 && ndb == 2) {
                    return matmul_2d(a_ptr, b_ptr, axis);
                }
                throw(std::runtime_error("'vecdot' supported for two 1d or two 2d tensors only."));
            } else
                throw(std::runtime_error("'vecdot' supported for 2 double or float tensors only."));
        }

        template<typename T1, typename T2>
        static ptr_type vecdot_1d(const T1 & a, const T2 & b, int axis)
        {
            auto d = xt::linalg::dot(a, b)();
            theTransceiver->reduce_all(&d, DTYPE<decltype(d)>::value, 1, SUM);
            return operatorx<decltype(d)>::mk_tx(d, REPLICATED);
        }

        template<typename A, typename B>
        static ptr_type matmul_2d(const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr, int axis)
        {
            if(a_ptr->slice().split_dim() != 0)
                throw(std::runtime_error("vecdoc_2d supported for split_dim=0 only"));

            auto nr = theTransceiver->nranks();
            auto me = theTransceiver->rank();
            rank_type right = (me + 1) % nr;
            rank_type left = (nr + me - 1) % nr;
            auto tsz = b_ptr->slice().tile_size(0);
            auto my_tshp_a = a_ptr->slice().tile_shape(me);
            auto tshp_b = b_ptr->slice().tile_shape(0);
            auto my_tshp_b = me == 0 ? tshp_b : b_ptr->slice().tile_shape(me);

            const auto & ax = a_ptr->xarray();
            const auto & bx = b_ptr->xarray();
            xt::xarray<A> cx = xt::zeros<A>({my_tshp_a[0], tshp_b[1]});
            auto buff = xt::empty<B>(tshp_b);
            if(tshp_b[0] == my_tshp_b[0]) {
                buff = bx;
            } else { // last partitions can be smaller -> need a view to assign values
                auto bv = xt::view(buff, xt::range(0, my_tshp_b[0]), xt::range(0, my_tshp_b[1]));
                bv = bx;
            }

            // We use an algo similar to Canon's
            // the last partitions can be smaller -> k depends on "me", the source rank of current partition
            for(rank_type i = nr; i>0; --i) {
                if(my_tshp_a[0]) {
                    TGEMM<A>::tgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    my_tshp_a[0], tshp_b[1], me == 0 ? tshp_b[0] : b_ptr->slice().tile_shape(me)[0],
                                    1, // alpha
                                    ax.data() + (me * tshp_b[0]),
                                    my_tshp_a[1], // lda
                                    buff.data(),
                                    tshp_b[1], // ldb
                                    1, // beta
                                    cx.data(),
                                    tshp_b[1]); // ldc
                }

                if(i > 1) {
                    // data exchange
                    // FIXME: optimize data transfer: last partition might contain unused data
                    theTransceiver->send_recv(buff.data(),
                                              tsz,
                                              DTYPE<A>::value,
                                              left,
                                              right);
                    me = (me + 1) % nr;
                }
            }
            return operatorx<A>::mk_tx(std::move(PVSlice({a_ptr->shape()[0], b_ptr->shape()[1]})), cx);
        }
    };
}

struct DeferredLinAlgOp : public Deferred
{
    tensor_i::future_type _a;
    tensor_i::future_type _b;
    int _axis;

    DeferredLinAlgOp(tensor_i::future_type & a, tensor_i::future_type & b, int axis)
        : _a(a), _b(b), _axis(axis)
    {}

    void run()
    {
        auto a = std::move(_a.get());
        auto b = std::move(_b.get());
        set_value(TypeDispatch<x::LinAlgOp>(a, b, _axis));
    }
};

tensor_i::future_type LinAlgOp::vecdot(tensor_i::future_type & a, tensor_i::future_type & b, int axis)
{
    return defer<DeferredLinAlgOp>(a, b, axis);
}
