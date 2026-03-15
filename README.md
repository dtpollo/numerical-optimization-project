# Unconstrained Optimization Methods

Implementation and comparison of three classical unconstrained 
optimization methods applied to two test functions.

## Methods
- Steepest Descent
- Conjugate Gradient
- Newton's Method

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Project Structure
```
├── Functions.py          # objective functions, gradients, and Hessians
├── Backtracking.py       # Armijo backtracking line search
├── Steepest_Method.py    # Steepest Descent algorithm
├── Conjugate_Method.py   # Conjugate Gradient algorithm
├── Newton_Method.py      # Newton's Method algorithm
├── Metrics.py            # convergence rate and computational cost
├── Plotting.py           # all plots and visualizations
└── main.py               # entry point — run this file
```

## How to Run

1. Clone the repository
```bash
git clone https://github.com/dtpollo/numerical-optimization-project
cd numerical-optimization-project
```

2. Open `main.py` and select which function to minimize:
```python
number_function = 1   # 1 or 2
x0 = np.array([2.0, 2.0])  # initial point
```

3. Run:
```bash
python main.py
```

## Test Functions

**Function 1:**
$$f_1(x_1, x_2) = 5x_1 + \frac{100}{x_1 x_2} + 2x_2$$
> Domain restriction: x1 > 0 and x2 > 0

**Function 2:**
$$f_2(x_1, x_2) = (x_1 - 5)^2 + x_1 x_2 + (x_2 - 2)^2$$

## Output

Running `main.py` produces:
- convergence history plot — f(x) and ||grad|| per iteration
- execution time comparison
- number of iterations comparison  
- convergence rate plot
- computational cost comparison

## Parameters

All methods share the same default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tol` | `1e-6` | convergence tolerance — stops when \|\|grad\|\| < tol |
| `max_iterations` | `10000` | maximum number of iterations |
| `alpha` | `1.0` | initial step size for backtracking |
| `rho` | `0.5` | reduction factor for backtracking |
| `c` | `1e-4` | Armijo constant |
