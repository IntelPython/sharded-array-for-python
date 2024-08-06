import sharpy as sp

sp.init(False)

a = sp.arange(0, 10, 1, sp.int64)
b = sp.reshape(a, (2, 5))
# print(b)
c = sp.permute_dims(b, [1, 0])
print(c)

# b = sp.arange(0, 100, 10, sp.int64)
# #print(b.dtype) # should _not_ trigger compilation
# c = a * b
# #print(c)
# d = sp.sum(c, [0])
# #del b          # generated function should _not_ return b
# print(a, c, d) # printing of c (not a!) should trigger compilation

sp.fini()
