import numpy as np

# Finds alpha satisfying the Armijo condition.
def backtracking(f, grad, x, direction, alpha=1.0, rho=0.5, c=1e-4):
    while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad, direction):
        alpha *= rho
    return alpha