#!/usr/bin/env python3
"""
LAB F8: Gradient descent. The algorithm that trains every model.

Now you put it together and MINIMIZE a function. Start somewhere, compute the
gradient (which way is uphill), step a little the opposite way, repeat. Watch the
value fall toward the minimum. This is the exact loop that trains an LLM, and it
is the bridge into Chapter 8 of the LLM course, where the same loop trains a real
model.

Run: python3 modules/academy-content/labs/foundations/f8-gradient-descent.py
"""
import sys

# minimize f(x) = (x - 3)^2. The minimum is obviously at x = 3, where f = 0.
# f'(x) = 2 * (x - 3).  We will let gradient descent DISCOVER x = 3 on its own.
def f(x):     return (x - 3) ** 2
def grad(x):  return 2 * (x - 3)

x = 0.0          # start far from the answer
lr = 0.1         # learning rate: how big a step to take
history = []
for step in range(50):
    loss = f(x)
    history.append((step, x, loss))
    x = x - lr * grad(x)   # step downhill

print("STEP 1-2: gradient descent minimizing f(x) = (x - 3)^2, starting at x = 0")
for step, xv, loss in [history[i] for i in (0, 5, 10, 20, 49)]:
    print(f"  step {step:2d}   x = {xv:6.4f}   loss = {loss:8.5f}")

# STEP 3: the invariant. Descent must converge: x ends near 3 and loss near 0,
# and the loss must be lower at the end than at the start (it actually learned).
final_x, final_loss = x, f(x)
ok = (abs(final_x - 3.0) < 0.01) and (final_loss < history[0][2])
print("")
print(f"STEP 3: converged to x ~ 3 and loss fell from {history[0][2]:.3f} to {final_loss:.5f}: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("That is training. In the LLM course you run this exact loop on a real model.")
print("You now have the Python and the math. You are ready to build LLMs.")
