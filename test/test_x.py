import ddptensor._ddptensor as dt
a = dt.Creator.create_from_shape(dt.ONES, [4,4], dt.float64)
b = dt.Creator.create_from_shape(dt.ONES, [4,4], dt.float64)
dt.IEWBinOp.op(dt.IADD, a, b)
print(a)
print(dt.EWBinOp.op(dt.EQ, a, b))
dt.fini()
