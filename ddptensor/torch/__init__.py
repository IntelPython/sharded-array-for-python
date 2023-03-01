from .. import dist_tensor as _cdt
import dtensor


def manual_seed(s=None):
    dtensor.__impl.manual_seed(s + _cdt.myrank() if s else _cdt.myrank())


def _rand_torch(shape, *args, **kwargs):
    return dtensor.__impl.rand(tuple(shape))


def rand(shape, *args, **kwargs):
    return dtensor.dtensor(
        _cdt.create(shape, "_rand_torch", "dtensor.torch", *args, **kwargs)
    )


def erf(ary):
    return dtensor.dtensor(_cdt.ew_unary_op(ary._t, "erf", False))
