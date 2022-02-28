import numpy as np
import ddptensor as dt
a = dt.ones((8,8), dtype=dt.float64)
b = dt.ones((8,8), dtype=dt.float64)
#s = dt.sum(a, [0,1])
sa = dt.sum(a, [1])
sb = dt.sum(b, [1])
print("s2:", sa, sb)
d = dt.vecdot(sa, sb, 0)
print(d)
if 0:
    b = dt.ones((8,8), dtype=dt.float64)
    c = a + b
    c = dt.zeros((8,8), dtype=dt.float64)
    ##print(b, c)
    b[0:8, 0:8] = c[0:8, 0:8]
    b[2:6, 0:8] = c[3:7, 0:8]
    b[3:8:2, 0:8] = c[3:6, 0:8]
    b[3:8:2, 0:8] = c[3:6, 0:8]
    #print(float(s))
    print(float(a[1,1]))
    print(float(b[0,7]))
    print(float(b[7,7]))
    print(float(b[7,6]))
    print(float(b[7,5]))
print("done")
dt.fini()
