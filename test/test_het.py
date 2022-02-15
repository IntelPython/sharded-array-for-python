import ddptensor as dt
a = dt.ones([8,8], dt.int16)
b = dt.zeros([8,8], dt.float64)
c = a*b
print(c)
dt.fini()
