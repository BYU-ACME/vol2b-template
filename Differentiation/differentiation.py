# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""

import time
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

from jax import numpy as jnp
from jax import grad, jacfwd, jacrev, jacobian, hessian, jit


# Problem 1
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    raise NotImplementedError("Problem 1 Incomplete")

def prob1():
    """
    Compare several finite difference formulas for f'(x).

    Returns
    -------
    None
        Plots the derivative for each difference method.
    """
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2(x0):
    """
    Compare the accuracy of several finite difference formulas for f'(x) at x0.

    Parameters
    ----------
    x0 : float
        The point at which to approximate the derivative.

    Returns
    -------
    None
        Plots absolute errors on a log-log scale for each difference method.
    """
    # Define the function and its exact derivative.
    f = lambda x: (np.sin(x) + 1) ** np.sin(np.cos(x))
    exact = lambda x: (-(np.sin(x) + 1) * np.log(np.sin(x) + 1) * np.sin(x) * np.cos(np.cos(x)) + np.sin(np.cos(x)) * np.cos(x)) * (np.sin(x) + 1) ** (np.sin(np.cos(x)) - 1)
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def jacobian_cdq2(f, x, h=1e-5):
    """
    Approximate the Jacobian matrix of f : R^n -> R^m at x, using centered second order differences.

    Parameters
    ----------
    f : callable
        Multivariate function f: (n,) -> (m,).
    x : (n,) ndarray
        Point at which to evaluate the Jacobian.
    h : float, optional
        Step size for finite differences.

    Returns
    -------
    (m, n) ndarray
        Approximate Jacobian matrix of f at x.
    """
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    raise NotImplementedError("Problem 4 Incomplete")

def prob4():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def trapezoid(f, a, b, N):
    """Approximate the integral of a function f over an interval [a, b].

    Parameters:
        f (function): the function over which to take the integral.
        a (float): the left boundary of the interval.
        b (float): the right boundary of the interval.
        N (integer): the number of subintervals to use.
    
    Returns:
        (float) the approximate area under the curve.
    """
    raise NotImplementedError("Problem 5 Incomplete")

def prob5():
    """Use JAX and trapezoid() to approximate I'(b), and
       plot the error."""
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """
    Use JAX to compute the Jacobian of a function using jacfwd() and jacrev(),
    and compare the results with a finite difference approximation.

    First, use the test function from Problem 3 to compute its Jacobian using
    jacfwd() and jacrev(). Verify that both methods agree with each other and
    closely match the finite difference approximation.

    Next, compare the computation time of jacfwd() and jacrev() for functions
    f and g. Return the four times in the order
    (jacfwd(f), jacrev(f), jacfwd(g), jacrev(g)).
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7():
    """
    Use JAX to compute the Hessian of a scalar-valued function.

    Define g(x, y) = x^2 + xy + y^2. Compute the Hessian using
    jax.hessian(g) and using jacobian(grad(g)) at (2, 3). Verify that
    both methods produce the same result and return the result.
    """
    raise NotImplementedError("Problem 7 Incomplete")


# Problem 8
def prob8(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0 uniformly from [0, 1).
        2. Use sympy_derivative() to calculate the "exact" value of f′(x0).
            Time the entire process, including constructing the symbolic
            derivative on every iteration.
        3. Time how long it takes to approximate f'(x0) using cdq4(). Record
            the absolute error of the approximation.
        4. Before the loop, create and warm up a JIT-compiled JAX derivative.
            Time only its evaluation on each x0, using block_until_ready(),
            and record the absolute error.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """

    # Helper function for SymPy differentiation
    def sympy_derivative():
        """
        Return a callable function for the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy.

        Returns
        -------
        function
            Callable derivative as a function of x, compatible with NumPy.
        """
        # Define variable x and expression
        x = sy.symbols('x')
        expr = (sy.sin(x) + 1) ** sy.sin(sy.cos(x))
        
        # Take derivative and return
        deriv = sy.diff(expr, x)
        return sy.lambdify(x, deriv, "numpy")

    raise NotImplementedError("Problem 8 Incomplete")