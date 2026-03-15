import numpy as np
from Steepest_Method import steepest
from Conjugate_Method import conjugate
from Newton_Method import newton
import Plotting as pl
import Metrics as mt


if __name__ == "__main__":

    # ── Configuration ────────────────────────────────────────
    # Select function 1 or 2
    number_function = 1
    x0              = np.array([2.0, 2.0])
    tol             = 1e-6
    max_iterations  = 10000

    # ── Run all methods ──────────────────────────────────────
    methods = [steepest, conjugate, newton]
    results = []

    for method in methods:
        x_opt, f_opt, iterations, history, elapsed = method(
            number_function, x0, tol, max_iterations
        )
        results.append({
            "method"    : method.__name__,
            "x_opt"     : x_opt,
            "f_opt"     : f_opt,
            "iterations": iterations,
            "elapsed"   : elapsed,
            "history"   : history
        })

    # ── Compute metrics ──────────────────────────────────────
    cost = mt.computational_cost(results)

    # ── Print results ────────────────────────────────────────
    print("\n===== METHOD RESULTS =====\n")

    for r, c in zip(results, cost):

        ratios   = mt.convergence_rate(r["x_opt"], r["history"])
        avg_rate = sum(ratios) / len(ratios) if ratios else float('nan')

        print(f"  Method               : {r['method']}")
        print(f"  Minimizer x*         : {r['x_opt']}")
        print(f"  f(x*)                : {r['f_opt']:.8f}")
        print(f"  Iterations           : {r['iterations']}")
        print(f"  Avg convergence rate : {avg_rate:.6f}")
        print(f"  Time                 : {r['elapsed']:.6f} seconds")
        print(f"  Gradient evaluations : {c['grad_evals']}")
        print(f"  Hessian evaluations  : {c['hess_evals']}")
        print(f"  {'-'*40}")

    # ── Individual history plots (uncomment to use) ──────────
    #pl.plot_individual_history(results[0]["history"], "steepest")
    #pl.plot_individual_history(results[1]["history"], "conjugate")
    #pl.plot_individual_history(results[2]["history"], "newton")

    # ── Comparison plots ─────────────────────────────────────

    # f(x) and ||grad|| convergence for all methods
    pl.plot_general_history(results)

    # execution time per method
    pl.plot_times(results)

    # number of iterations per method
    pl.plot_iterations(results)

    # convergence rate r_k = ||e_{k+1}|| / ||e_k|| per method
    pl.plot_convergence_rate(results, [r["x_opt"] for r in results])

    # gradient and hessian evaluations per method
    pl.plot_cost(cost)