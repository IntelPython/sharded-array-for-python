import ddptensor as dt
a = dt.arange(1, 25, 1, dtype=dt.float64)
b = dt.arange(1, 31, 1, dtype=dt.float64)
print("a", a)
print("b", b)
a = dt.reshape(a, (4,6))
b = dt.reshape(b, (6,5))
print("a", a)
print("b", b)
print()
c = dt.vecdot(a, b, 0)
print(c)

import numpy as np
a = np.arange(1, 25, 1, dtype=np.float64)
b = np.arange(1, 31, 1, dtype=np.float64)
a = np.reshape(a, (4,6))
b = np.reshape(b, (6,5))
c = np.dot(a, b)
print(c)

dt.fini()
