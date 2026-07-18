#!/usr/bin/env python3
"""
LAB F4: Vectors. The math object at the center of machine learning.

A vector is just a list of numbers, but with operations that mean something:
adding two vectors, scaling one, and the dot product, which measures how aligned
two vectors are. Embeddings, attention, and every layer of a model are vector
math. You implement the operations here, by hand, and prove their properties.

Run: python3 modules/academy-content/labs/foundations/f4-vectors.py
"""
import sys, math

def add(a, b):     return [x + y for x, y in zip(a, b)]
def scale(a, k):   return [x * k for x in a]
def dot(a, b):     return sum(x * y for x, y in zip(a, b))
def length(a):     return math.sqrt(dot(a, a))

u = [1, 2, 3]
v = [4, 5, 6]
print("STEP 1: vectors are lists of numbers")
print(f"  u = {u}")
print(f"  v = {v}")

print("")
print("STEP 2: the operations")
print(f"  u + v      = {add(u, v)}")
print(f"  3 * u      = {scale(u, 3)}")
print(f"  u . v      = {dot(u, v)}   (dot product: bigger means more aligned)")
print(f"  |u|        = {length(u):.3f}   (length)")

# STEP 3: invariants that MUST hold for real vector math:
#  (a) the dot product is commutative: u.v == v.u
#  (b) a vector dotted with itself equals its length squared
ok = (dot(u, v) == dot(v, u)) and (abs(dot(u, u) - length(u) ** 2) < 1e-9)
print("")
print(f"STEP 3: u.v == v.u, and u.u == |u|^2: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Vectors are how models hold meaning. Next: matrices, vectors stacked into grids.")
