import ddptensor as dt
a = dt.ones([4,4], dt.float64)
b = dt.ones([4,4], dt.float64)
a += b
print(a)
print(a == b)
print(dt.sqrt(a))
print(dt.sum(a, [1]))
dt.fini()
