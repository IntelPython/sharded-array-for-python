import ddptensor as dt
aa = dt.ones([8,8], dt.float64)
a = aa[0:8:2, 0:8:2]
b = dt.zeros([4,4], dt.float64)
a[1:4,0:3] = b[0:3, 0:3]
print(a)
print(aa)
print(type(aa), aa.dtype, aa.shape, aa.size, int(aa[0:1,0:1]), len(aa))
dt.fini()
