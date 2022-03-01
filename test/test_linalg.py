import ddptensor as dt
a = dt.arange(1, 36, 1, dtype=dt.float64)
b = dt.arange(1, 22, 1, dtype=dt.float64)
print("a", a)
print("b", b)
a = dt.reshape(a, (5,7))
b = dt.reshape(b, (7,3))
print("a", a)
print("b", b)
print()
c = dt.vecdot(a, b, 0)
print(c)

import numpy as np
a = np.arange(1, 36, 1, dtype=np.float64)
b = np.arange(1, 22, 1, dtype=np.float64)
a = np.reshape(a, (5,7))
b = np.reshape(b, (7,3))
c = np.dot(a, b)
print(c)

dt.fini()
