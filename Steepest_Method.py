import Functions as fn
import Backtracking as bk
import numpy as np
import time

def steepest(number_function, x0, tol=1e-6, max_iterations=10000):
    x = x0.copy()
    history = []

    if number_function == 1:
        f, grad_f = fn.f1, fn.grad_f1
    elif number_function == 2:
        f, grad_f = fn.f2, fn.grad_f2
    else:
        print("Just enter 1 or 2 as number FUNCTION")
        exit()

    start = time.time()

    for k in range(max_iterations):

        # Compute gradient and it norm at current point
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        # Save current state
        history.append({
            "iteration" : k,
            "x"         : x.copy(),
            "f(x)"      : f(x),
            "||grad||"  : grad_norm
        })

        # Check convergence
        if grad_norm < tol:
            break

        # Search direction (negative gradient)
        direction = -grad

        # Compute step size with backtracking
        alpha = bk.backtracking(f, grad, x, direction)

        # Update point
        x = x + alpha * direction

    elapsed = time.time() - start

    return x, f(x), k, history, elapsed
