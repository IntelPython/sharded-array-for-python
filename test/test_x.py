import ddptensor._ddptensor as dt
a = dt.Creator.create_from_shape(dt.ONES, [4,4], dt.float64)
b = dt.Creator.create_from_shape(dt.ONES, [4,4], dt.float64)
dt.IEWBinOp.op(dt.__IADD__, a, b)
print(a)
print(dt.EWBinOp.op(dt.EQUAL, a, b))
print(dt.EWUnyOp.op(dt.SQRT, a))
dt.fini()
