from .. import dist_tensor as _cdt
from .. import __impl
from ..dtensor import dtensor


def seed(s=None):
    __impl.random.seed(s + _cdt.myrank() if s else _cdt.myrank())


def _uniform_numpy(shape, start, stop, dtype=None):
    return __impl.random.uniform(start, stop, shape)


def uniform(start, stop, shape):
    return dtensor(
        _cdt.create(shape, "_uniform_numpy", "dtensor.numpy.random", start, stop)
    )

    #    for func in ["seed", "uniform"]:


#        exec(
#            f"{func} = staticmethod(lambda shape, *args, **kwargs: dtensor(_cdt.create('{func}', _cdt.__dlp_provider_name + '.random', *args, **kwargs)))"
#        )
