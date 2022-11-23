// SPDX-License-Identifier: BSD-3-Clause

#include "ddptensor/EWBinOp.hpp"
#include "ddptensor/LinAlgOp.hpp"
#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/Creator.hpp"
#include "ddptensor/TypePromotion.hpp"
#include "ddptensor/CollComm.hpp"
#include "ddptensor/DDPTensorImpl.hpp"

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Shape/IR/Shape.h>

// #######################################################################################
// The 2 operators/tensors can have shifted partitions, e.g. local data might not be the
// same on a and b. This means we
// need to copy/communicate to bring the relevant parts of the tensor to the
// right processes.
//
// The minimal copy would be to map one array to other. This however can create
// a unbalanced distribution which we currently only support through slicing.
// We could create larger-than-necessary tiles and slice, but that can easily
// require much more memory than we need. Hence we map both to a newly created,
// equally tiled array (e.g. the result).
//
// In principle each input tensor has 3 regions for mapping to the resulting
// partitioning. Each region can have any size between 0 and the full span.
//   1. Leading data at the beginning of the partition which comes from remote
//   2. local data which does not need communication
//   3. Trailing remote data
//
// We attempt to minimize copies by treating each region explicitly, e.g. data
// which is already local will not be copied or communicated.
//
// Additionally, to reduce generated code size we convert buffers to the result
// type before applying to XTensor. This means when type conversion is needed
// we'll use the Transceiver/MPI to convert and do a copy for local data.
//
// We create a buffer and fetch the remote data to it. We then sweep over the
// above regions and apply the op. All buffers/views get linearized/flattened.
// #######################################################################################

#if 0
namespace x {

    // @return true if operation returs bool, false otherwise
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

        template<typename T, typename AB>
        struct mapped
        {
            const std::shared_ptr<DPTensorX<T>> & _sptr;
            // is a replicated/scalar?
            bool _isrepl;
            // can we use local data of a directly?
            bool _opt;
            // receive buffer
            std::vector<AB> _rbuff;
            // local buffers to store data if we need to copy (convert type)
            std::vector<AB> _ary;
            // generic pointers to apply to xt::adapt, will point to comm buffer, local data or local buffer
            const AB * _ptr;
            // slice of local tile
            NDSlice _lslc;
            // pointer to start of needed local data of a
            const T * _ldata;
            // region info
            uint64_t _reg[3];

            mapped(const std::shared_ptr<DPTensorX<T>> & sptr, PVSlice & r_slc, uint64_t tsz, rank_type rank)
                : _sptr(sptr),
                  _rbuff(),
                  _ary(),
                  _ptr(nullptr)
            {
                // global size
                uint64_t _sz(_sptr->slice().size());
                // compute overlap with result
                std::array<std::vector<NDSlice>, 2> _overlap(r_slc.map_ranks(_sptr->slice()));
                _isrepl = _overlap[1].empty();
                _opt = std::is_same<T, AB>::value && _sptr->slice().local_is_contiguous();
                // Fetch the remote parts and get displacement info for a: [0] for sending, [1] for receiving
                std::array<uint64_t, 2> _displ(CollComm::coll_copy(_sptr, _overlap, _rbuff));
                // size of needed local part
                uint64_t _lsz(_isrepl ? _sz : _overlap[1][rank].size());
                // size of remote part
                uint64_t _buffsz(_rbuff.size());
                // slice of local tile of a
                _lslc = _isrepl ? NDSlice(shape_type({_sz})) : PVSlice(_sptr->slice(), _overlap[0][rank]).tile_slice();
                _ldata = _sptr->xarray().data() + _displ[0] + *_sptr->slice().tile_slice().begin();
                _reg[0] = _isrepl ? 0 : _displ[1];          // end leading buffer
                _reg[1] = _isrepl ? tsz : _displ[1] + _lsz; // end local data
                _reg[2] = tsz;
            }

            const AB * get_ptr(uint64_t c_pos, uint64_t p_pos, uint64_t & csz) {
                csz = _isrepl ? 1 : c_pos - p_pos;
                if(_isrepl && p_pos > 0) return _ptr;
                if(c_pos <= _reg[0]) {
                    // This is within leading buffer a
                    return (_ptr = _rbuff.data() + p_pos);
                }
                if(c_pos <= _reg[1]) {
                    // This is within a's local data
                    uint64_t off = p_pos - _reg[0];
                    if(_opt) {
                        // start of local buffer a, contiguous and of type AB
                        return _ptr = (reinterpret_cast<const AB*>(_ldata + off));
                    } else {
                        uint64_t ncopied;
                        // start of local buffer a, non-contiguous and/or not of type AB
                        auto it = (_lslc.begin() += off);
                        _ary.resize(csz);
                        it.fill_buffer(_ary.data(), csz, ncopied, _sptr->xarray().data());
                        return (_ptr = _ary.data());
                    }
                }
                // trailing buffer
                uint64_t off = _reg[0] + p_pos - _reg[1];
                return (_ptr = _rbuff.data() + off);
            }
        };
            
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

            rank_type rank = getTransceiver()->rank();
            // Slice for/of the result
            PVSlice r_slc(a_sptr->slice().size() >= b_sptr->slice().size() ? a_sptr->slice().shape() : b_sptr->slice().shape());
            // Size of local result tile
            uint64_t tsz = r_slc.local_size();
            // allocate result
            auto r_x = xt::empty<R>(std::move(r_slc.tile_shape()));
            auto r_xv = xt::reshape_view(r_x, typename xt::xarray<R>::shape_type({tsz}));
            R * ptr = reinterpret_cast<R*>(r_x.data());

            mapped<A, AB> a_map(a_sptr, r_slc, tsz, rank);
            mapped<B, AB> b_map(b_sptr, r_slc, tsz, rank);

            int a_i = 0, b_i = 0;
            uint64_t p_pos = 0, pos = 0;

            while(a_i < 2 || b_i < 2) {
                uint64_t a_csz = 0, b_csz = 0;
                uint64_t ancopied, bncopied;

                uint64_t c_pos = std::min(a_map._reg[a_i], b_map._reg[b_i]);

                if(p_pos == c_pos) {
                    if(c_pos == a_map._reg[a_i] && a_i < 2) ++a_i;
                    if(c_pos == b_map._reg[b_i] && b_i < 2) ++b_i;
                    continue;
                }
                assert(c_pos > p_pos);

                auto a_ptr = a_map.get_ptr(c_pos, p_pos, a_csz);
                auto b_ptr = b_map.get_ptr(c_pos, p_pos, b_csz);

                if(a_csz && b_csz) {
                    uint64_t r_csz = std::max(a_csz, b_csz);
                    auto a_view = xt::adapt(a_ptr, a_csz, xt::no_ownership(), typename xt::xarray<AB>::shape_type({a_csz}));
                    auto b_view = xt::adapt(b_ptr, b_csz, xt::no_ownership(), typename xt::xarray<AB>::shape_type({b_csz}));
                    auto r_view = xt::view(r_xv, xt::range(pos, pos + r_csz));
                    
                    dispatch_op<R, AB>(bop, r_view, a_view, b_view);
                    
                    ptr += r_csz;
                    pos += r_csz;
                }
                p_pos = c_pos;
                assert(p_pos == pos);
                if(c_pos == a_map._reg[a_i] && a_i < 2) ++a_i;
                if(c_pos == b_map._reg[b_i] && b_i < 2) ++b_i;
            }
            return operatorx<R>::mk_tx(std::move(r_slc), std::move(r_x));
        }

        template<typename R, typename AB, typename RX, typename SX, typename std::enable_if<std::is_same<R, bool>::value, bool>::type = true>
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
            default:
                if constexpr (std::is_integral<AB>::value) {
                    switch(bop) {
                    case LOGICAL_XOR:
                        rv = ((bv || av) && xt::not_equal(xt::cast<bool>(av), xt::cast<bool>(bv)));
                        break;
                    default:
                        throw std::runtime_error("Unknown/invalid boolean elementwise binary operation");
                    }
                } else std::runtime_error("Unknown/invalid boolean elementwise binary operation");
            }
        }

        template<typename R, typename AB, typename RX, typename SX, typename std::enable_if<!std::is_same<R, bool>::value, bool>::type = true>
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
            default:
                if constexpr (std::is_integral<AB>::value) {
                    switch(bop) {
                    case __LSHIFT__:
                    case BITWISE_LEFT_SHIFT:
                        rv = av << bv;
                        break;
                    case __MOD__:
                    case REMAINDER:
                        rv = av % bv;
                        break;
                    case __RSHIFT__:
                    case BITWISE_RIGHT_SHIFT:
                        rv = av >> bv;
                        break;
                    case __AND__:
                    case BITWISE_AND:
                        rv = av & bv;
                        break;
                    case __OR__:
                    case BITWISE_OR:
                        rv = av | bv;
                        break;
                    case __XOR__:
                    case BITWISE_XOR:
                        rv = av ^ bv;
                        break;
                    case __RAND__:
                        rv = bv & av;
                        break;
                    case __ROR__:
                        rv = bv | av;
                        break;
                    case __RXOR__:
                        rv = bv ^ av;
                        break;
                    case __RLSHIFT__:
                        rv = bv << av;
                        break;
                    case __RMOD__:
                        rv = bv % av;
                        break;
                    case __RRSHIFT__:
                        rv = bv >> av;
                        break;
                    default:
                        throw std::runtime_error("Unknown/invalid elementwise binary operation");
                    }
                } else throw std::runtime_error("Unknown/invalid elementwise binary operation");
            }
        }
    };
} // namespace x
#endif // if 0

// convert id of our binop to id of imex::ptensor binop
static ::imex::ptensor::EWBinOpId ddpt2mlir(const EWBinOpId bop)
{
    switch(bop) {
    case __ADD__:
    case ADD:
    case __RADD__:
        return ::imex::ptensor::ADD;
    case ATAN2:
        return ::imex::ptensor::ATAN2;
    case __FLOORDIV__:
    case FLOOR_DIVIDE:
    case __RFLOORDIV__:
        return ::imex::ptensor::FLOOR_DIVIDE;
        // __MATMUL__ is handled before dispatching, see below
    case __MUL__:
    case MULTIPLY:
    case __RMUL__:
        return ::imex::ptensor::MULTIPLY;
    case __SUB__:
    case SUBTRACT:
    case __RSUB__:
        return ::imex::ptensor::SUBTRACT;
    case __TRUEDIV__:
    case DIVIDE:
    case __RTRUEDIV__:
        return ::imex::ptensor::TRUE_DIVIDE;
    case __POW__:
    case POW:
    case __RPOW__:
        return ::imex::ptensor::POWER;
    case LOGADDEXP:
        return ::imex::ptensor::LOGADDEXP;
    case __LSHIFT__:
    case BITWISE_LEFT_SHIFT:
    case __RLSHIFT__:
        return ::imex::ptensor::BITWISE_LEFT_SHIFT;
    case __MOD__:
    case REMAINDER:
    case __RMOD__:
        return ::imex::ptensor::MODULO;
    case __RSHIFT__:
    case BITWISE_RIGHT_SHIFT:
    case __RRSHIFT__:
        return ::imex::ptensor::BITWISE_RIGHT_SHIFT;
    case __AND__:
    case BITWISE_AND:
    case __RAND__:
        return ::imex::ptensor::BITWISE_AND;
    case __OR__:
    case BITWISE_OR:
    case __ROR__:
        return ::imex::ptensor::BITWISE_OR;
    case __XOR__:
    case BITWISE_XOR:
    case __RXOR__:
        return ::imex::ptensor::BITWISE_XOR;
    default:
        throw std::runtime_error("Unknown/invalid elementwise binary operation");
    }
}
#pragma GCC diagnostic pop

struct DeferredEWBinOp : public Deferred
{
    id_type _a;
    id_type _b;
    EWBinOpId _op;

    DeferredEWBinOp() = default;
    DeferredEWBinOp(EWBinOpId op, const tensor_i::future_type & a, const tensor_i::future_type & b)
        : Deferred(a.dtype(), a.rank()),
          _a(a.id()), _b(b.id()), _op(op)
    {}

    void run() override
    {
#if 0
        const auto a = std::move(Registry::get(_a).get());
        const auto b = std::move(Registry::get(_b).get());
        set_value(std::move(TypeDispatch<x::EWBinOp>(a, b, _op)));
#endif
    }

    bool generate_mlir(::mlir::OpBuilder & builder, ::mlir::Location loc, jit::DepManager & dm) override
    {
        // FIXME the type of the result is based on a only
        auto av = dm.getDependent(builder, _a);
        auto bv = dm.getDependent(builder, _b);

        auto aPtTyp = ::imex::dist::getPTensorType(av);
        assert(aPtTyp);

        dm.addVal(this->guid(),
                  builder.create<::imex::ptensor::EWBinOp>(loc, aPtTyp, builder.getI32IntegerAttr(ddpt2mlir(_op)), av, bv),
                  [this](uint64_t rank, void *allocated, void *aligned, intptr_t offset, const intptr_t * sizes, const intptr_t * strides,
                         uint64_t * gs_allocated, uint64_t * gs_aligned, uint64_t * lo_allocated, uint64_t * lo_aligned) {
            this->set_value(std::move(mk_tnsr(_dtype, rank, allocated, aligned, offset, sizes, strides,
                                              gs_allocated, gs_aligned, lo_allocated, lo_aligned)));
        });
        return false;
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
