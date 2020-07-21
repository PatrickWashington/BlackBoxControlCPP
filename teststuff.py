import numpy as np

A = np.random.random((2,5))
print(A)

# B = np.linalg.inv(A.T @ A) @ A.T
# print(B)

# I = B @ A
# print(I)

Ai = np.linalg.pinv(A)
print(Ai)

print(Ai @ A)

x = np.random.random((A.shape[1],1))
print(x)
print((Ai@A)@x)