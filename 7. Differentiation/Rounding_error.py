import numpy as np

# rounding error(round-off error)
def numerical_diff_(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) /h

print(np.float32(1e-50))
