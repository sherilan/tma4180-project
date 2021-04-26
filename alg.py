
import warnings
import numpy as np
import pandas as pd


EPS = 1e-6

def objective(A, v, x):
    """
    Evaluates the median_l2 objective

    Args:
        A (np.array<m,2>): Location of m points in R^2
        v (np.array<m>): Objective weight of each point
        x (np.array<?,2>): Gradient evaluation point(s)

    Returns:
        The median_l2 objective value as an <?> float array
    """
    dist = np.linalg.norm(x[...,None,:] - A, ord=2, axis=-1)
    return (v * dist).sum(axis=-1)

def gradient(A, v, x, eps=EPS):
    """
    Computes the gradient of the median_l2 objective

    Args:
        A (np.array<m,2>): Location of m points in R^2
        v (np.array<m>): Objective weight of each point
        x (np.array<2>): Gradient evaluation point

    Returns:
        An np.array<2> with the gradient at point x
    """
    # Compute vectors A[i] -> x and their L2 norm
    diff = x - A
    norm = np.linalg.norm(diff, ord=2, axis=-1)
    # Handle values too close to any A
    too_small = norm < eps
    if too_small.any():
        warnings.warn('Omitting gradient from too close point')
    # Compute unit diffs (which are the unweighted grad. contribs. from each A[i])
    unit_diff = diff[~too_small] / norm[~too_small, None]
    # Multiply in weights v_i and aggregate contributions across all A[i]
    return (v[~too_small, None] * unit_diff).sum(axis=0)

def objective_upper_bound(A, v, x):
    """
    Computes the upper bound UB(x) >= |f(x*) - f(x)|
    """
    grad_norm = np.linalg.norm(gradient(A, v, x), ord=2)
    sigma = np.linalg.norm(x - A, ord=2).max()
    return sigma * grad_norm

def stopping_criterion_1(A, v, epsilon):
    """
    Generates stopping signal if LB(x) > 0 and UB(x) / LB(x) < epsilon
    """
    def criterion_1(x):
        UB = objective_upper_bound(A, v, x)
        LB = objective(A, v, x) - UB
        if LB <= 0:
            return False, np.nan
        else:
            bound = UB / LB
            return bound < epsilon, bound
    return criterion_1

def stopping_criterion_2(A, v, epsilon):
    """
    Generates sstopping signal if UB(x) / f(x) < epsilon
    """
    def criterion_2(x):
        UB = objective_upper_bound(A, v, x)
        bound = UB / objective(A, v, x)
        return bound < epsilon, bound
    return criterion_2


def test_k(A, v, k):
     # Generate index set M with i=k subtracted
    M = [i for i in range(len(A)) if i != k]
    # Compute gradient contributions for all i != k
    diff = A[k] - A[M]
    norm = np.linalg.norm(diff, ord=2, axis=-1)
    # Multiply in weights v_i and sum all gradient contributions
    grad = (v[M, None] * diff / norm[:,None]).sum(axis=0)
    # And take its L2 norm
    return np.linalg.norm(grad, ord=2)

def check_initial_solution(A, v):
    for k in range(len(A)):
        if test_k(A, v, k) <= v[k]:
            return k, A[k], objective(A, v, A[k])

def initialize_com(A, v, seed=None):
    """
    Generates an initial iterate with the center-of-mass method
    """
    return np.average(A, weights=v, axis=0)

def initialize_bbox(A, v, seed=None):
    """
    Generates an initial iterate within the bounding box of A
    """
    minv = A.min(axis=0)
    maxv = A.max(axis=0)
    return initialize_uniform(minv=minv, maxv=maxv, seed=seed)

def initialize_uniform(minv=(-1, -1), maxv=(1, 1), seed=None):
    """
    Generates an initial iterate by sampling a uniform distribution
    """
    random = np.random.RandomState(seed)
    return random.uniform(minv, maxv)

def weiszfeld_update(A, v, x, eps=EPS):
    """
    Performs one Weiszfeld Algorithm update on x
    """
    # Compute distances
    norms = np.linalg.norm(A - x, ord=2, axis=-1)
    # Check if x is too close to any of the points in A
    too_close = norms < eps
    if too_close.any():
        warnings.warn('Norm too close to zero in Weiszfeld update')
    # Compute weights
    weights = v[~too_close] / norms[~too_close]
    # Compute next iterate
    return (weights[:,None] * A[~too_close]).sum(axis=0) / weights.sum()

def weiszfeld(A, v, x0, criterion=None, max_iters=1000):
    """
    Runs the weizefeld algorithm for the l2 median problem
    """
    history = []
    x = x0

    for i in range(max_iters):
        f_x = objective(A, v, x)
        stop, bound = criterion(x) if criterion else (False, np.nan)
        x_new = weiszfeld_update(A, v, x)
        history.append(dict(x0=x[0], x1=x[1], f_x=f_x, bound=bound, stop=stop))
        if stop:
            break
        else:
            x = x_new
    else:
        if criterion:
            warnings.warn('Weiszfeld algorithm did not terminate!')

    return pd.DataFrame(history)

def backtracking_line_search(A, v, x, p, alpha, rho, c, max_iters=100):
    """
    Performs standard backtracking line search

    Args:
        A (np.array<m,2>): Location of m points in R^2
        v (np.array<m>): Objective weight of each point
        x (np.array<2>): Current iterate
        p (np.array<2>): Line search direction
        alpha (float): Maximum (initial) step length
        rho (float):
        c (float):
        max_iters (int): Safeguard against infinite looping

    Returns:
        An np.array<2> with the next iterate

    """
    f = lambda x: objective(A, v, x)
    grad_f = lambda x: gradient(A, v ,x)
    a = alpha
    for i in range(max_iters):
        f_cur = f(x + a * p)
        f_tar = f(x) + c * a * np.dot(grad_f(x), p)  # Inefficient
        if f_cur <= f_tar:
            break
        else:
            a = a * rho
    else:
        warnings.warn(f'Exceeded max_iters ({max_iters}) in line search')
        a = 0  # Don't change the iterate at all since we failed to improve

    return i + 1, x + a * p

def gradient_descent(
    A, v, x0, criterion=None, alpha=1.0, rho=0.5, c=0.5, max_iters=1000
):
    """
    Runs gradient descent with backtracking line search
    """
    history = []
    x = x0

    for i in range(max_iters):
        f_x = objective(A, v, x)
        nabla_f_x = gradient(A, v, x)
        stop, bound = criterion(x) if criterion else (False, np.nan)
        steps, x_new = backtracking_line_search(
            A, v, x, p=-nabla_f_x, alpha=alpha, rho=rho, c=c
        )
        history.append(
            dict(x0=x[0], x1=x[1], f_x=f_x, bound=bound, stop=stop, steps=steps)
        )
        if stop:
            break
        else:
            x = x_new
    else:
        if criterion:
            warnings.warn('Gradient descent did not terminate!')

    return pd.DataFrame(history)
