#include <xtensor/xview.hpp>
using namespace xt::placeholders;
#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/TypePromotion.hpp"
#include "ddptensor/CollComm.hpp"
#include "ddptensor/Chunker.hpp"

// #######################################################################################
// The minimal copy would be to map one array to other. This however can create
// a unbalanced distribution which we currently only support through slicing.
// We could create larger-than-necessary tiles and slice, but that can easily
// require much more memory than we need.
// Hence we map both to a newly created, euqually tiled array.

// With 2 distributed arrays a and b, th overlap of their local data with the local data
// of the result can vary. At any place of the local result, the data comes from either
// remote (_) or local (a or b). Notice, neither a or b nor result and any of a or need
// to have aligned paritions. This leads to 5 possible regions in ther resulting partition:
//
// _______aaaaaaaaa____
// ___bbbbbbbb_________
// r1 |r2 |r3 |r4 |r5
//
//   r1 has data from remote a and remote b
//   r2 is when the start of local a and local b are shifted
//   r3 is where all data comes from local a and b
//   ...
// In some scenarios some regions might be empty (like when partitions of and b have aligned starts or ends or both).
//
// We create a buffer and fetch the remote data to it. We then apply the operator to each of the above region spearately.
// All buffers/views get linearized/flattened.
// #######################################################################################

namespace x {
    
#define def_op(_name, _op)                  \
    struct op_##_name {                          \
        template<typename A, typename B>     \
        auto operator()(A && a, B && b) {        \
            return _op(std::move(a), std::move(b)); \
        }                                        \
    };
    struct op_EQUAL {
        template<typename A, typename B>
        auto operator()(A && a, B && b) {
            return xt::equal(std::move(a), std::move(b));
        }
    };

    def_op(ADD, xt::operator+);
#define __radd(_a, _b) (_b) + (_a)
    def_op(RADD, __radd);
    def_op(ATAN2, xt::atan2);
    //def_op(EQUAL, xt::equal(a, b));
#define __flordiv(_a, _b) xt::floor(a / b)
    def_op(FLOOR_DIVIDE, __flordiv);
    def_op(GREATER_EQUAL, xt::operator>=);
    def_op(GREATER, xt::operator>);
    def_op(LESS_EQUAL, xt::operator<=);
    def_op(LESS, xt::operator<);
    // __MATMUL__ is handled before dispatching, see below
    def_op(MULTIPLY, xt::operator*);
#define __rmul(_a, _b) (_b) * (_a)
    def_op(RMULTIPLY, __rmul);
    def_op(NOT_EQUAL, xt::not_equal);
    def_op(SUBTRACT, xt::operator-);
    def_op(DIVIDE, xt::operator/);
#define __rdiv(_a, _b) (_b) / (_a)
    def_op(RDIVIDE, __rdiv);
#define __rflordiv(_a, _b) xt::floor(_b / _a)
    def_op(RFLOORDIV, __rflordiv);
#define __rsub(_a, _b) (_b) - (_a)
    def_op(RSUB, __rsub);
    def_op(POW, xt::pow);
#define __rpow(_a, _b) xt::pow(_b, _a)
    def_op(RPOW, __rpow);
#define __logaddexp(_a, _b) xt::log(xt::exp(_a) + xt::exp(_b))
    def_op(LOGADDEXP, __logaddexp);
#define __logical_and(_a, _b) xt::cast<bool>(_a) && xt::cast<bool>(_b)
    def_op(LOGICAL_AND, __logical_and);
#define __logical_or(_a, _b) xt::cast<bool>(_a) || xt::cast<bool>(_b)
    def_op(LOGICAL_OR, __logical_or);
#define __logical_xor(_a, _b) ((_b || _a) && xt::not_equal(xt::cast<bool>(_a), xt::cast<bool>(_b)))
    def_op(LOGICAL_XOR, __logical_xor);
    def_op(BITWISE_AND, xt::operator&);
#define __rbitwise_and(_a, _b) (_b) & (_a)
    def_op(RBITWISE_AND, __rbitwise_and);
    def_op(BITWISE_LEFT_SHIFT, xt::operator<<);
    def_op(REMAINDER, xt::operator%);
    def_op(BITWISE_OR, xt::operator|);
#define __rbitwise_or(_a, _b) (_b) | (_a)
    def_op(RBITWISE_OR, __rbitwise_or);
    def_op(BITWISE_RIGHT_SHIFT, xt::operator>>);
    def_op(BITWISE_XOR, xt::operator^);
#define __rbitwise_xor(_a, _b) (_b) ^ (_a)
    def_op(RXOR, __rbitwise_xor);
#define __rlshift(_a, _b) (_b) << (_a)
    def_op(RLSHIFT, __rlshift);
#define __rmod(_a, _b) (_b) % (_a)
    def_op(RMOD, __rmod);
#define __rrshift(_a, _b) (_b) >> (_a)
    def_op(RRSHIFT, __rrshift);

    bool is_bool(EWBinOpId bop)
    {
        switch(bop) {
        case __EQ__:
        case __GE__:
        case __GT__:
        case __LE__:
        case __LT__:
        case __NE__:
        case EQUAL:
        case GREATER:
        case GREATER_EQUAL:
        case LESS_EQUAL:
        case LOGICAL_AND:
        case LOGICAL_OR:
        case LOGICAL_XOR:
        case LESS:
        case NOT_EQUAL:
            return true;
        }
        return false;
    }
            
    class EWBinOp
    {
    public:
        using ptr_type = DPTensorBaseX::ptr_type;

#pragma GCC diagnostic ignored "-Wswitch"
        template<typename A, typename B>
        static ptr_type op(EWBinOpId bop, const std::shared_ptr<DPTensorX<A>> & a_ptr, const std::shared_ptr<DPTensorX<B>> & b_ptr)
        {
            if(is_bool(bop)) return do_op<A, B, bool>(bop, a_ptr, b_ptr);
            return do_op<A, B, typename promote<A, B>::type>(bop, a_ptr, b_ptr);
        }

        // FIXME broadcastable tensors not supported except scalars
        template<typename A, typename B, typename R>
        static ptr_type do_op(EWBinOpId bop, const std::shared_ptr<DPTensorX<A>> & a_sptr, const std::shared_ptr<DPTensorX<B>> & b_sptr)
        {
            using AB = typename promote<A, B>::type;
            // Step 1: Get the mapping of a and b to our resulting slice

            uint64_t a_sz = a_sptr->slice().size();
            uint64_t b_sz = b_sptr->slice().size();
            assert(a_sz > 0 && b_sz > 0);

            // Slice for/of the result
            PVSlice r_slc(a_sz >= b_sz ? a_sptr->slice().shape() : b_sptr->slice().shape());
            // Get the overlap for a_sptr;
            auto a_overlap = r_slc.map_ranks(a_sptr->slice());
            // Get the overlap for a_sptr;
            auto b_overlap = r_slc.map_ranks(b_sptr->slice());

            bool a_isrepl = a_overlap[1].empty();
            bool b_isrepl = b_overlap[1].empty();

            // Step 2: Fetch the remote needed data of a and b
            
            rank_type rank = theTransceiver->rank();
            std::vector<AB> rbuffa, rbuffb;
            auto a_displ = CollComm::coll_copy(a_sptr, a_overlap, rbuffa);
            auto b_displ = CollComm::coll_copy(b_sptr, b_overlap, rbuffb);
            
            uint64_t a_lsz = a_isrepl ? a_sz : a_overlap[1][rank].size(); // size of needed local part of a
            uint64_t b_lsz = b_isrepl ? b_sz : b_overlap[1][rank].size(); // size of needed local part of b
            uint64_t a_buffsz = rbuffa.size();         // size of remote part of a
            uint64_t b_buffsz = rbuffb.size();         // size of remote part of b
            
            // Step 3: apply op to region

            std::vector<AB> a_ary;
            std::vector<AB> b_ary;
            const AB * a_ptr = nullptr;
            const AB * b_ptr = nullptr;
            NDSlice a_lslc = a_isrepl ? NDSlice(shape_type({a_sz})) : PVSlice(a_sptr->slice(), a_overlap[0][rank]).tile_slice();
            NDSlice b_lslc = b_isrepl ? NDSlice(shape_type({b_sz})) : PVSlice(b_sptr->slice(), b_overlap[0][rank]).tile_slice();
            auto a_off = *a_sptr->slice().tile_slice().begin();
            const A * a_ldata = a_sptr->xarray().data() + a_displ[0] + a_off;
            auto b_off = *b_sptr->slice().tile_slice().begin();
            const B * b_ldata = b_sptr->xarray().data() + b_displ[0] + b_off;
            bool a_opt = std::is_same<A, AB>::value && a_sptr->slice().local_is_contiguous();
            bool b_opt = std::is_same<B, AB>::value && b_sptr->slice().local_is_contiguous();
            
            // allocate result
            uint64_t tsz = r_slc.local_size();
            auto r_x = xt::empty<R>(std::move(r_slc.tile_shape()));
            auto r_xv = xt::reshape_view(r_x, typename xt::xarray<R>::shape_type({tsz}));
            R * ptr = reinterpret_cast<R*>(r_x.data());
            uint64_t pos = 0;

            uint64_t a_reg[] = {
                a_isrepl ? 0 : a_displ[1],         // end leading buffer
                a_isrepl ? tsz : a_displ[1] + a_lsz, // end local data
                tsz
            };
            uint64_t b_reg[] = {
                b_isrepl ? 0 : b_displ[1],         // end leading buffer
                b_isrepl ? tsz : b_displ[1] + b_lsz, // end local data
                tsz
            };

            int a_i = 0, b_i = 0;
            uint64_t p_pos = 0;

            auto get_ptr = [](uint64_t c_pos, uint64_t p_pos, bool isrepl, auto reg, bool opt, std::vector<AB> & rbuff,
                              auto * ldata, const NDSlice & lslc, std::vector<AB> & ary, uint64_t & csz, auto sptr) -> const AB * {
                csz = isrepl ? 1 : c_pos - p_pos;
                if(c_pos <= reg[0]) {
                    // This is within leading buffer a
                    return rbuff.data() + p_pos;
                }
                if(c_pos <= reg[1]) {
                    // This is within a's local data
                    uint64_t _off = p_pos - reg[0];
                    if(opt) {
                        // start of local buffer a, contiguous and of type AB
                        return reinterpret_cast<const AB*>(ldata + _off);
                    } else {
                        uint64_t ncopied;
                        // start of local buffer a, non-contiguous and/or not of type AB
                        auto it = (lslc.begin() += _off);
                        ary.resize(csz);
                        it.fill_buffer(ary.data(), csz, ncopied, sptr->xarray().data());
                        return ary.data();
                    }
                }
                // trailing buffer
                uint64_t _off = reg[0] + p_pos - reg[1];
                return rbuff.data() + _off;
            };
            
            while(a_i < 2 || b_i < 2) {
                uint64_t a_csz = 0, b_csz = 0;
                uint64_t ancopied, bncopied;

                uint64_t c_pos = std::min(a_reg[a_i], b_reg[b_i]);

                if(p_pos == c_pos) {
                    if(c_pos == a_reg[a_i] && a_i < 2) ++a_i;
                    if(c_pos == b_reg[b_i] && b_i < 2) ++b_i;
                    continue;
                }
                assert(c_pos > p_pos);

                if(! a_isrepl || p_pos == 0) a_ptr = get_ptr(c_pos, p_pos, a_isrepl, a_reg, a_opt, rbuffa, a_ldata, a_lslc, a_ary, a_csz, a_sptr);
                if(! b_isrepl || p_pos == 0) b_ptr = get_ptr(c_pos, p_pos, b_isrepl, b_reg, b_opt, rbuffb, b_ldata, b_lslc, b_ary, b_csz, b_sptr);

                if(a_csz && b_csz) {
                    uint64_t r_csz = std::max(a_csz, b_csz);
                    auto a_view = xt::adapt(a_ptr, a_csz, xt::no_ownership(), typename xt::xarray<AB>::shape_type({a_csz}));
                    auto b_view = xt::adapt(b_ptr, b_csz, xt::no_ownership(), typename xt::xarray<AB>::shape_type({b_csz}));
                    auto r_view = xt::view(r_xv, xt::range(pos, pos + r_csz));
                    
                    dispatch_op<R>(bop, r_view, a_view, b_view);
                    
                    ptr += r_csz;
                    pos += r_csz;
                }
                p_pos = c_pos;
                assert(p_pos == pos);
                if(c_pos == a_reg[a_i] && a_i < 2) ++a_i;
                if(c_pos == b_reg[b_i] && b_i < 2) ++b_i;
            }
            return operatorx<R>::mk_tx(std::move(r_slc), std::move(r_x));
        }

        template<typename R, typename RX, typename SX, typename std::enable_if<std::is_same<R, bool>::value, bool>::type = true>
        static void dispatch_op(EWBinOpId bop, RX && rv, SX && av, SX && bv)
        {
            switch(bop) {
            case __EQ__:
            case EQUAL:
                rv = xt::equal(av, bv);
                break;
            case __GE__:
            case GREATER_EQUAL:
                rv = av >= bv;
                break;
            case __GT__:
            case GREATER:
                rv = av > bv;
                break;
            case __LE__:
            case LESS_EQUAL:
                rv = av <= bv;
                break;
            case __LT__:
            case LESS:
                rv = av < bv;
                break;
            case __NE__:
            case NOT_EQUAL:
                rv = xt::not_equal(av, bv);
                break;
            case LOGICAL_AND:
                rv = xt::cast<bool>(av) && xt::cast<bool>(bv);
                break;
            case LOGICAL_OR:
                rv = xt::cast<bool>(av) || xt::cast<bool>(bv);
                break;
            case LOGICAL_XOR:
                rv = ((bv || av) && xt::not_equal(xt::cast<bool>(av), xt::cast<bool>(bv)));
                break;
            default:
                throw std::runtime_error("Unknown/invalid boolean elementwise binary operation");
            }
        }

        template<typename R, typename RX, typename SX, typename std::enable_if<!std::is_same<R, bool>::value, bool>::type = true>
        static void dispatch_op(EWBinOpId bop, RX && rv, SX && av, SX && bv)
        {
            switch(bop) {
            case __ADD__:
            case ADD:
                rv = av + bv;
                break;
            case __RADD__:
                rv = bv + av;
                break;
            case ATAN2:
                rv = xt::atan2(av, bv);
                break;
            case __FLOORDIV__:
            case FLOOR_DIVIDE:
                rv = xt::floor(av / bv);
                break;
                // __MATMUL__ is handled before dispatching, see below
            case __MUL__:
            case MULTIPLY:
                rv = av * bv;
                break;
            case __RMUL__:
                rv = bv * av;
                break;
            case __SUB__:
            case SUBTRACT:
                rv = av - bv;
                break;
            case __TRUEDIV__:
            case DIVIDE:
                rv = av / bv;
                break;
            case __RFLOORDIV__:
                rv = xt::floor(bv / av);
                break;
            case __RSUB__:
                rv = bv - av;
                break;
            case __RTRUEDIV__:
                rv = bv / av;
                break;
            case __POW__:
            case POW:
                rv = xt::pow(av, bv);
                break;
            case __RPOW__:
                rv = xt::pow(bv, av);
                break;
            case LOGADDEXP:
                rv = xt::log(xt::exp(av) + xt::exp(bv));
                break;
            };
#if 0
            if constexpr (std::is_integral<A>::value && std::is_integral<typename T2::value_type>::value) {
                switch(bop) {
                case __AND__:
                case BITWISE_AND:
                    return operatorx<A>::mk_tx_(a_ptr, a & b);
                case __RAND__:
                    return operatorx<A>::mk_tx_(a_ptr, b & a);
                case __LSHIFT__:
                case BITWISE_LEFT_SHIFT:
                    return operatorx<A>::mk_tx_(a_ptr, a << b);
                case __MOD__:
                case REMAINDER:
                    return operatorx<A>::mk_tx_(a_ptr, a % b);
                case __OR__:
                case BITWISE_OR:
                    return operatorx<A>::mk_tx_(a_ptr, a | b);
                case __ROR__:
                    return operatorx<A>::mk_tx_(a_ptr, b | a);
                case __RSHIFT__:
                case BITWISE_RIGHT_SHIFT:
                    return operatorx<A>::mk_tx_(a_ptr, a >> b);
                case __XOR__:
                case BITWISE_XOR:
                    return operatorx<A>::mk_tx_(a_ptr, a ^ b);
                case __RXOR__:
                    return operatorx<A>::mk_tx_(a_ptr, b ^ a);
                case __RLSHIFT__:
                    return operatorx<A>::mk_tx_(a_ptr, b << a);
                case __RMOD__:
                    return operatorx<A>::mk_tx_(a_ptr, b % a);
                case __RRSHIFT__:
                    return operatorx<A>::mk_tx_(a_ptr, b >> a);
                }
                default:
                    throw std::runtime_error("Unknown/invalid elementwise binary operation");
            } else throw std::runtime_error("Unknown/invalid elementwise binary operation");
#endif
        }
#pragma GCC diagnostic pop

    };
} // namespace x

struct DeferredEWBinOp : public Deferred
{
    id_type _a;
    id_type _b;
    EWBinOpId _op;

    DeferredEWBinOp() = default;
    DeferredEWBinOp(EWBinOpId op, const tensor_i::future_type & a, const tensor_i::future_type & b)
        : _a(a.id()), _b(b.id()), _op(op)
    {}

    void run()
    {
        const auto a = std::move(Registry::get(_a).get());
        const auto b = std::move(Registry::get(_b).get());
        set_value(std::move(TypeDispatch<x::EWBinOp>(a, b, _op)));
    }

    FactoryId factory() const
    {
        return F_EWBINOP;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template value<sizeof(_b)>(_b);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * EWBinOp::op(EWBinOpId op, const ddptensor & a, const py::object & b)
{
    auto bb = Creator::mk_future(b);
    if(op == __MATMUL__) {
        return LinAlgOp::vecdot(a, *bb, 0);
    }
    return new ddptensor(defer<DeferredEWBinOp>(op, a.get(), bb->get()));
}

FACTORY_INIT(DeferredEWBinOp, F_EWBINOP);

#if 0

        template<typename OP, typename A, typename B>
        static ptr_type do_split_op(OP && op,
                                    const A & a, const B & b,
                                    const shape_type & gshape, const PVSlice & res_slc, 
                                    const Buffer & rbuffa, const Buffer & rbuffb,
                                    uint64_t abuffsz, uint64_t bbuffsz, uint64_t asz,
                                    const std::array<uint64_t, 2> & adispl, const std::array<uint64_t, 2> & bdispl,
                                    const std::array<uint64_t, 5> & regs)
        {
            return do_split_op_<typename promote<typename A::value_type, typename B::value_type>::type>
                (std::move(op), a, b, gshape, res_slc, rbuffa, rbuffb, abuffsz, bbuffsz, asz, adispl, bdispl, regs);
        }

        template<typename R, typename OP, typename A, typename B>
        static ptr_type do_split_op_(OP && op,
                                     const A & a, const B & b,
                                     const shape_type & gshape, const PVSlice & res_slc, 
                                     const Buffer & rbuffa, const Buffer & rbuffb,
                                     uint64_t abuffsz, uint64_t bbuffsz, uint64_t asz,
                                     const std::array<uint64_t, 2> & adispl, const std::array<uint64_t, 2> & bdispl,
                                     const std::array<uint64_t, 5> & regs)
        {
            // allocate result
            typename xt::xarray<R>::shape_type shpx = {res_slc.local_size()};
            auto res_x = xt::empty<R>(shpx);

            if(regs[0]) {
                // both from buffer
                auto av = xt::adapt(rbuffa, {regs[0]});
                auto bv = xt::adapt(rbuffb, {regs[0]});
                auto rv = xt::view(res_x, xt::range(_, regs[0]));
                rv = op(std::move(av), std::move(bv));
            } // else nothing in any buffer prepending local data
            uint64_t aoff = 0, boff = 0;
            if(regs[1] > regs[0]) {
                uint64_t _cnt = regs[1] - regs[0];
                if(regs[1] == adispl[1]) {
                    // b comes from local, a from buffer
                    const auto _a = reinterpret_cast<const typename A::value_type*>(rbuffa.data()); // raw pointer to buffer data of remote a
                    auto av = xt::adapt(_a+regs[0], {_cnt});
                    auto bv = xt::view(xt::reshape_view(b, {b.size()}), xt::range(bdispl[0], bdispl[0] + _cnt));
                    auto rv = xt::view(res_x, xt::range(regs[0], regs[1]));
                    rv = op(std::move(av), std::move(bv));
                    boff = _cnt;
                } else {
                    // a comes from local, b from buffer
                    const auto _b = reinterpret_cast<const typename B::value_type*>(rbuffb.data()); // raw pointer to buffer data of remote b
                    auto av = xt::view(xt::reshape_view(a, {a.size()}), xt::range(adispl[0], adispl[0] + _cnt));
                    auto bv = xt::adapt(_b+regs[0], {_cnt});
                    auto rv = xt::view(res_x, xt::range(regs[0], regs[1]));
                    rv = op(std::move(av), std::move(bv));
                    aoff = _cnt;
                }
            } // else both buffers have same amount of prepending data
            if(regs[2] > regs[1]) {
                // both come from local
                uint64_t _cnt = regs[2] - regs[1];
                auto av = xt::view(xt::reshape_view(a, {a.size()}), xt::range(adispl[0] + aoff, adispl[0] + aoff + _cnt));
                auto bv = xt::view(xt::reshape_view(b, {b.size()}), xt::range(bdispl[0] + boff, bdispl[0] + boff + _cnt));
                auto rv = xt::view(res_x, xt::range(regs[1], regs[2]));
                rv = op(std::move(av), std::move(bv));
                aoff += _cnt;
                boff += _cnt;
            } // else no overlap in local data
            if(regs[3] > regs[2]) {
                uint64_t _cnt = regs[3] - regs[2];
                if(regs[3] == adispl[1] + asz) {
                    // a from local, b from buffer
                    const auto _b = reinterpret_cast<const typename B::value_type*>(rbuffb.data()); // raw pointer to buffer data of remote b
                    auto av = xt::view(xt::reshape_view(a, {a.size()}), xt::range(adispl[0] + aoff, adispl[0] + aoff + _cnt));
                    auto bv = xt::adapt(_b + bdispl[1], {_cnt});
                    auto rv = xt::view(res_x, xt::range(regs[2], regs[3]));
                    rv = op(std::move(av), std::move(bv));
                } else {
                    // b from local, a from buffer
                    const auto _a = reinterpret_cast<const typename A::value_type*>(rbuffa.data()); // raw pointer to buffer data of remote a
                    auto av = xt::adapt(_a + adispl[1], {_cnt});
                    auto bv = xt::view(xt::reshape_view(b, {b.size()}), xt::range(bdispl[0] + boff, bdispl[0] + boff + _cnt));
                    auto rv = xt::view(res_x, xt::range(regs[2], regs[3]));
                    rv = op(std::move(av), std::move(bv));
                }
            } // local data of a an b end equally
            if(regs[4] > regs[3]) {
                // both from buffer
                uint64_t _cnt = regs[4] - regs[3];
                const auto _a = reinterpret_cast<const typename A::value_type*>(rbuffa.data()); // raw pointer to buffer data of remote a
                const auto _b = reinterpret_cast<const typename B::value_type*>(rbuffb.data()); // raw pointer to buffer data of remote b
                auto av = xt::adapt(_a + abuffsz - _cnt, {_cnt});
                auto bv = xt::adapt(_b + bbuffsz - _cnt, {_cnt});
                auto rv = xt::view(res_x, xt::range(regs[3], regs[4]));
                rv = op(std::move(av), std::move(bv));
            } // data data after last local
                    
            res_x.reshape(res_slc.local_shape());
            return operatorx<R>::mk_tx(gshape, res_x);
        }

#endif
