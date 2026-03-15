import numpy as np


def convergence_rate(x_optimum, history):
    ratios = []
    for i in range(1, len(history)):
        xk      = history[i]["x"]
        xk_prev = history[i-1]["x"]

        ek      = np.linalg.norm(xk - x_optimum)
        ek_prev = np.linalg.norm(xk_prev - x_optimum)

        if ek_prev > 1e-10:
            ratios.append(ek / ek_prev)

    return ratios


def computational_cost(results):
    cost = []
    for r in results:
        iterations  = r["iterations"]
        grad_evals  = len(r["history"])
        hess_evals  = len(r["history"]) if r["method"] == "newton" else 0

        cost.append({
            "method"     : r["method"],
            "iterations" : iterations,
            "grad_evals" : grad_evals,
            "hess_evals" : hess_evals
        })

    return cost
