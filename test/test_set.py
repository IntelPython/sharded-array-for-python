# import ddptensor as dt
# dt.init(False)
# aa = dt.ones([8,8], dt.float64)
# a = aa[0:8:2, 0:8:2]
# b = dt.zeros([4,4], dt.float64)
# a[1:4,0:3] = b[0:3, 0:3]
# print(a)
# print(aa)
# print(type(aa), aa.dtype, aa.shape, aa.size, int(aa[0:1,0:1]), len(aa))
# if 1:
#     b = dt.ones((8,8), dtype=dt.float64)
#     c = a + b
#     c = dt.zeros((8,8), dtype=dt.float64)
#     ##print(b, c)
#     b[0:8, 0:8] = c[0:8, 0:8]
#     b[2:6, 0:8] = c[3:7, 0:8]
#     b[3:8:2, 0:8] = c[3:6, 0:8]
#     b[3:8:2, 0:8] = c[3:6, 0:8]
#     #print(float(s))
#     print(float(a[1:2,1:2]))
#     print(float(b[0:1,7:8]))
#     print(float(b[7:8,7:8]))
#     print(float(b[7:8,6:7]))
#     print(float(b[7:8,5:6]))
# dt.fini()
