import numpy as np

import sharpy as sp


def sp_tranposed3d_1():
    a = sp.arange(0, 2 * 3 * 4, 1)
    a = sp.reshape(a, [2, 3, 4])

    # b = a.swapaxes(1,0).swapaxes(1,2)
    b = sp.permute_dims(a, (1, 0, 2))  # 2x4x4 -> 4x2x4 || 4x4x4
    b = sp.permute_dims(b, (0, 2, 1))  # 4x2x4 -> 4x4x2 || 4x4x4

    # c = b.swapaxes(1,2).swapaxes(1,0)
    c = sp.permute_dims(b, (0, 2, 1))
    c = sp.permute_dims(c, (1, 0, 2))

    assert np.allclose(sp.to_numpy(a), sp.to_numpy(c))
    return b


def sp_tranposed3d_2():
    a = sp.arange(0, 2 * 3 * 4, 1)
    a = sp.reshape(a, [2, 3, 4])

    # b = a.swapaxes(2,1).swapaxes(2,0)
    b = sp.permute_dims(a, (0, 2, 1))
    b = sp.permute_dims(b, (2, 1, 0))

    # c = b.swapaxes(2,1).swapaxes(0,1)
    c = sp.permute_dims(b, (0, 2, 1))
    c = sp.permute_dims(c, (1, 0, 2))

    return c


def np_tranposed3d_1():
    a = np.arange(0, 2 * 3 * 4, 1)
    a = np.reshape(a, [2, 3, 4])
    b = a.swapaxes(1, 0).swapaxes(1, 2)
    return b


def np_tranposed3d_2():
    a = np.arange(0, 2 * 3 * 4, 1)
    a = np.reshape(a, [2, 3, 4])
    b = a.swapaxes(2, 1).swapaxes(2, 0)
    c = b.swapaxes(2, 1).swapaxes(0, 1)
    return c


sp.init(False)

b1 = sp_tranposed3d_1()
assert np.allclose(sp.to_numpy(b1), np_tranposed3d_1())

b2 = sp_tranposed3d_2()
assert np.allclose(sp.to_numpy(b2), np_tranposed3d_2())

sp.fini()
