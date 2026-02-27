from black_scholes_merton import price_option_bsm
from binomial_model import price_option_tree
from greeks import vega_bsm

import numpy as np

def implied_vol_Newton(option_type, exercise_style, price, s, k, r, T, q,
                       sigma0=0.1, tol=1e-8, max_iter=500):
    """
    Compute the implied volatility of a European call option or a European put option using the Newton-Raphson method.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        exercise_style (string): must be 'European'
        price (float): market option price
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (years)
        q (float): dividend yield
        sigma0 (float): initial volatility guess
        tol (float): tolerance for price difference
        max_iter (int): maximum number of iterations

    Returns:
        float: Implied volatility
    """

    if exercise_style != 'European':
        raise ValueError("Newton method implemented only for European options.")

    sigma = sigma0

    for _ in range(max_iter):

        price_bsm = price_option_bsm(option_type=option_type, exercise_style=exercise_style, 
                                     s=s, k=k, r=r, sigma=sigma, T=T, q=q)
        vega = vega_bsm(s, k, r, sigma, T, q)
        diff = price_bsm - price

        # check the convergence
        if abs(diff)<tol:
            return sigma

        # prevent division by very small vol
        if abs(vega)<1e-8:
            print("Warning: Vega too small, Newton-Raphson may not converge.")
            break
        sigma -= diff/vega

        # prevent negative volatility
        if sigma<=0: 
            sigma = 1e-4

    return sigma if abs(diff) < tol else None
