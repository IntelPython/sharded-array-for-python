import ddptensor as dt
a = dt.arange(4, 148, 1, dt.int64)
print(a)
b = dt.reshape(a, [3, 48])
print(b)
dt.fini()
