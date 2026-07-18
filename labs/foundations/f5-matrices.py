#!/usr/bin/env python3
"""
LAB F5: Matrices. Grids of numbers, and the multiply that powers every model.

A matrix is a list of rows, each row a vector. Matrix multiplication is THE core
operation of a neural network: every layer is a matrix multiply. You implement it
here by hand and prove it against the identity matrix, which leaves any matrix
unchanged, the matrix version of multiplying by 1.

Run: python3 modules/academy-content/labs/foundations/f5-matrices.py
"""
import sys

def matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]

def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print("STEP 1: matrices are lists of rows")
print(f"  A = {A}")
print(f"  B = {B}")

C = matmul(A, B)
print("")
print("STEP 2: matrix multiply, row of A dotted with column of B")
print(f"  A x B = {C}")

# STEP 3: the identity invariant. A x I == A and I x A == A, for any A.
I = identity(2)
ok = (matmul(A, I) == A) and (matmul(I, A) == A)
print("")
print(f"STEP 3: A x I == A and I x A == A (identity leaves A unchanged): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Every layer of a transformer is a matrix multiply. Next: probability and softmax.")
