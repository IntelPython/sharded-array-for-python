#include "ddptensor/IEWBinOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/DDPTensorImpl.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Creator.hpp"

#if 0
namespace x {

    class IEWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename A, typename B>
        static ptr_type op(IEWBinOpId iop, std::shared_ptr<DPTensorX<A>> a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            auto & ax = a_ptr->xarray();
            const auto & bx = b_ptr->xarray();
            if(a_ptr->is_sliced() || b_ptr->is_sliced()) {
                auto av = xt::strided_view(ax, a_ptr->lslice());
                const auto & bv = xt::strided_view(bx, b_ptr->lslice());
                return do_op(iop, av, bv, a_ptr);
            }
            return do_op(iop, ax, bx, a_ptr);
        }

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename A, typename T1, typename T2>
        static ptr_type do_op(IEWBinOpId iop, T1 & a, const T2 & b, std::shared_ptr<DPTensorX<A>> a_ptr)
        {
            switch(iop) {
            case __IADD__:
                a += b;
                return a_ptr;
            case __IFLOORDIV__:
                a = xt::floor(a / b);
                return a_ptr;
            case __IMUL__:
                a *= b;
                return a_ptr;
            case __ISUB__:
                a -= b;
                return a_ptr;
            case __ITRUEDIV__:
                a /= b;
                return a_ptr;
            case __IPOW__:
                throw std::runtime_error("Binary inplace operation not implemented");
            }
            if constexpr (std::is_integral<typename T1::value_type>::value && std::is_integral<typename T2::value_type>::value) {
                switch(iop) {
                case __IMOD__:
                    a %= b;
                    return a_ptr;
                case __IOR__:
                    a |= b;
                    return a_ptr;
                case __IAND__:
                    a &= b;
                    return a_ptr;
                case __IXOR__:
                    a ^= b;
                case __ILSHIFT__:
                    a = xt::left_shift(a, b);
                    return a_ptr;
                case __IRSHIFT__:
                    a = xt::right_shift(a, b);
                    return a_ptr;
                }
            }
            throw std::runtime_error("Unknown/invalid inplace elementwise binary operation");
        }
#pragma GCC diagnostic pop

    };
} // namespace x
#endif // if 0

struct DeferredIEWBinOp : public Deferred
{
    id_type _a;
    id_type _b;
    IEWBinOpId _op;

    DeferredIEWBinOp() = default;
    DeferredIEWBinOp(IEWBinOpId op, const tensor_i::future_type & a, const tensor_i::future_type & b)
        : _a(a.id()), _b(b.id()), _op(op)
    {}

    void run()
    {
        // const auto a = std::move(Registry::get(_a).get());
        // const auto b = std::move(Registry::get(_b).get());
        // set_value(std::move(TypeDispatch<x::IEWBinOp>(a, b, _op)));
    }
     
    FactoryId factory() const
    {
        return F_IEWBINOP;

    }
    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template value<sizeof(_b)>(_b);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * IEWBinOp::op(IEWBinOpId op, ddptensor & a, const py::object & b)
{
    auto bb = Creator::mk_future(b);
    auto res = new ddptensor(defer<DeferredIEWBinOp>(op, a.get(), bb.first->get()));
    if(bb.second) delete bb.first;
    return res;
}

FACTORY_INIT(DeferredIEWBinOp, F_IEWBINOP);
