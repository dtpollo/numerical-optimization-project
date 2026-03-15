import numpy as np


# ===================
# FUNCTIONS
# ===================

def f1(x):
    x1, x2 = x[0], x[1]
    return 5*x1 + 100/(x1*x2) + 2*x2


def f2(x):
    x1, x2 = x[0], x[1]
    return (x1 - 5)**2 + x1*x2 + (x2 - 2)**2


# ==================
# GRADIENTS
# ==================

def grad_f1(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 5 - 100 / (x1**2 * x2)
    df_dx2 = -100 / (x1 * x2**2) + 2
    return np.array([df_dx1, df_dx2])


def grad_f2(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 2*(x1 - 5) + x2
    df_dx2 = x1 + 2*(x2 - 2)
    return np.array([df_dx1, df_dx2])


# ===============
# HESSIANS 
# ===============

def hess_f1(x):
    x1, x2 = x[0], x[1]
    h11 = 200 / (x1**3 * x2)
    h22 = 200 / (x1 * x2**3)
    h12 = 100 / (x1**2 * x2**2)
    return np.array([[h11, h12],
                     [h12, h22]])


def hess_f2(x):
    return np.array([[2.0, 1.0],
                     [1.0, 2.0]])