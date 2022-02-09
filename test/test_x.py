import ddptensor as dt
a = dt.ones([4,4], dt.float64)
b = dt.ones([4,4], dt.float64)
a += b
print(a)
print(a == b)
print(dt.sqrt(a))
print(dt.sum(a, [1]))
print(a[0:1,0:1], float(a[0:1,0:1]), bool(a[0:1,0:1]), int(a[0:1,0:1]))
print(a[0:2,0:2])
dt.fini()
