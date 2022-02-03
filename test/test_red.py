import ddptensor as dt
a = dt.ones((8,8))
s = dt.sum(a, axis=None)
b = dt.ones((8,8))
c = a + b
c = dt.zeros((8,8))
##print(b, c)
b[0:8, 0:8] = c[0:8, 0:8]
b[2:6, 0:8] = c[3:7, 0:8]
b[3:8:2, 0:8] = c[3:6, 0:8]
b[3:8:2, 0:8] = c[3:6, 0:8]
print(float(s))
print(float(a[1,1]))
print(float(b[0,7]))
print(float(b[7,7]))
print(float(b[7,6]))
print(float(b[7,5]))
print("done")
dt.fini()
