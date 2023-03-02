import ddptensor as aa

# import numpy as aa
import numpy as np

aa.init(False)

n = 16
r = 2
A = aa.ones((n, n), dtype=aa.int64)
B = aa.zeros((n, n), dtype=aa.int64)
W = aa.ones(((2 * r + 1), (2 * r + 1)), dtype=aa.int64)
for i in range(3):
    #     if i:
    #         C = B[2:n-2,2:n-2]
    #         B[2:n-2,2:n-2] = C
    #     else:
    # D = B[2:n-2,2:n-2]
    B[2 : n - 2, 2 : n - 2] = (
        B[2 : n - 2, 2 : n - 2]
        + W[2, 2] * A[2 : n - 2, 2 : n - 2]
        + W[2, 0] * A[2 : n - 2, 0 : n - 4]
        + W[2, 1] * A[2 : n - 2, 1 : n - 3]
        + W[2, 3] * A[2 : n - 2, 3 : n - 1]
        + W[2, 4] * A[2 : n - 2, 4 : n - 0]
        + W[0, 2] * A[0 : n - 4, 2 : n - 2]
        + W[1, 2] * A[1 : n - 3, 2 : n - 2]
        + W[3, 2] * A[3 : n - 1, 2 : n - 2]
        + W[4, 2] * A[4 : n - 0, 2 : n - 2]
    )
    # A[2:n-2,2:n-2] \
    #    + A[2:n-2,2:n-2] \
    #     + A[2:n-2,0:n-4] \
    #     + A[2:n-2,1:n-3] \
    #     + A[2:n-2,3:n-1] \
    #     + A[2:n-2,4:n-0] \
    #     + A[0:n-4,2:n-2] \
    #     + A[1:n-3,2:n-2] \
    #     + A[3:n-1,2:n-2] \
    #     + A[4:n-0,2:n-2]
    A[0:n, 0:n] = A + 1
print(B)

aa.fini()
