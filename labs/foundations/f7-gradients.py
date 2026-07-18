#!/usr/bin/env python3
"""
LAB F7: Derivatives and gradients. Which way is downhill.

Training a model means changing its weights to lower a loss. To know WHICH way to
change them, you need the gradient: the slope of the loss. Here you compute a
derivative two ways, numerically (nudge the input and see how the output changes)
and analytically (the exact formula), and prove they agree. That agreement is the
foundation of every training loop.

Run: python3 modules/academy-content/labs/foundations/f7-gradients.py
"""
import sys

# f(x) = x^2. Its exact derivative is f'(x) = 2x (basic calculus).
def f(x):        return x * x
def f_exact(x):  return 2 * x

# numerical derivative: how much f changes for a tiny nudge h, divided by h.
def f_numeric(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

print("STEP 1-2: derivative of f(x) = x^2, numeric vs exact")
xs = [1.0, 2.0, 5.0, -3.0]
worst = 0.0
for x in xs:
    num, exact = f_numeric(x), f_exact(x)
    worst = max(worst, abs(num - exact))
    print(f"  x={x:5.1f}   numeric={num:8.4f}   exact={exact:8.4f}")

# STEP 3: the invariant. The numerical and exact gradients must agree closely.
# If they did not, we could not trust the gradients that train a model.
ok = worst < 1e-4
print("")
print(f"STEP 3: numeric and exact gradients agree (max error {worst:.2e}): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("The gradient points uphill, so we step the OTHER way. Next: gradient descent.")
