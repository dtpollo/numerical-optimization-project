import matplotlib.pyplot as plt
import numpy as np


def plot_individual_history(history, method_name="Method"):
    """Plots f(x) and ||grad|| for a single method."""

    iterations = [h["iteration"] for h in history]
    f_values   = [h["f(x)"]     for h in history]
    grad_norms = [h["||grad||"]  for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{method_name} — Convergence History", fontsize=13)

    ax1.plot(iterations, f_values, color="steelblue")
    ax1.set_title("f(x) vs Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)")

    ax2.plot(iterations, grad_norms, color="darkorange")
    ax2.set_title("||grad|| vs Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||grad||")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()


def plot_general_history(results):
    """Plots f(x) and ||grad|| for all methods on the same axes."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Convergence Comparison", fontsize=13)

    for r in results:
        history    = r["history"]
        label      = r["method"]
        iterations = [h["iteration"] for h in history]
        f_values   = [h["f(x)"]     for h in history]
        grad_norms = [h["||grad||"]  for h in history]

        ax1.plot(iterations, f_values,   label=label)
        ax2.plot(iterations, grad_norms, label=label)

    ax1.set_title("f(x) vs Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)")
    ax1.legend()

    ax2.set_title("||grad|| vs Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||grad||")
    ax2.set_yscale("log")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_times(results):
    """Bar chart comparing execution time of each method."""

    methods = [r["method"]  for r in results]
    times   = [r["elapsed"] for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, times, color=["steelblue", "darkorange", "seagreen"])

    # add value on top of each bar
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(times)*0.01,
                 f"{t:.4f}s", ha="center", fontsize=9)

    plt.title("Execution Time")
    plt.xlabel("Method")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


def plot_iterations(results):
    """Bar chart comparing number of iterations of each method."""

    methods    = [r["method"]     for r in results]
    iterations = [r["iterations"] for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, iterations,
                   color=["steelblue", "darkorange", "seagreen"])

    # add value on top of each bar
    for bar, it in zip(bars, iterations):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(iterations)*0.01,
                 str(it), ha="center", fontsize=9)

    plt.title("Number of Iterations by Method")
    plt.xlabel("Method")
    plt.ylabel("Iterations")
    plt.tight_layout()
    plt.show()


def plot_convergence_rate(results, x_optimums):
    """
    Plots the convergence rate r_k = ||e_{k+1}|| / ||e_k||
    for each method.

    Parameters:
        results    : list of dicts from main.py
        x_optimums : list of x* for each method (same order as results)
    """

    plt.figure(figsize=(8, 5))

    for r, x_opt in zip(results, x_optimums):
        history = r["history"]
        label   = r["method"]
        ratios  = []

        for i in range(1, len(history)):
            xk      = history[i]["x"]
            xk_prev = history[i-1]["x"]
            ek      = np.linalg.norm(xk - x_opt)
            ek_prev = np.linalg.norm(xk_prev - x_opt)
            if ek_prev > 1e-10:
                ratios.append(ek / ek_prev)

        plt.plot(range(len(ratios)), ratios, label=label)

    plt.title("Convergence Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Rate $r_k$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cost(cost):
    """
    Bar chart comparing gradient and Hessian evaluations per method.

    Parameters:
        cost : list of dicts from mt.computational_cost(results)
    """

    methods    = [c["method"]     for c in cost]
    grad_evals = [c["grad_evals"] for c in cost]
    hess_evals = [c["hess_evals"] for c in cost]

    x     = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, grad_evals, width,
           label="Gradient evals", color="steelblue")
    ax.bar(x + width/2, hess_evals, width,
           label="Hessian evals",  color="darkorange")

    ax.set_title("Computational Cost — Evaluations per Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Evaluations")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.tight_layout()
    plt.show()