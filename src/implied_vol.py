from black_scholes_merton import price_option_bsm
from binomial_model import price_option_tree
from greeks import vega_bsm

import numpy as np
from scipy import optimize

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

    if abs(diff) < tol:
        return sigma
    else: 
        print("Newton-Raphson did not converge, try Bisection") 
        implied_vol_bisect(option_type, exercise_style, price, s, k, r, T, q,
                           sigmaLow=1e-4, sigmaHigh=5, tol=1e-8, max_iter=500)

def implied_vol_bisect(option_type, exercise_style, price, s, k, r, T, q,
                       sigmaLow=1e-4, sigmaHigh=5, tol=1e-8, max_iter=500):
    """
    Compute the implied volatility of a European call option or a European put option using the Bisection method.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        exercise_style (string): must be 'European'
        price (float): market option price
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (years)
        q (float): dividend yield
        sigmaLow (float): initial volatility lower bound
        sigmaHigh (float): initial volatility upper bound
        tol (float): tolerance for price difference
        max_iter (int): maximum number of iterations

    Returns:
        float: Implied volatility
    """

    if exercise_style != 'European':
        raise ValueError("Bisection method implemented only for European options.")


    price_bsmLow = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                        s=s, k=k, r=r, sigma=sigmaLow, T=T, q=q)
    price_bsmHigh = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
	                                s=s, k=k, r=r, sigma=sigmaHigh, T=T, q=q)
    
    if(price_bsmLow - price) * (price_bsmHigh - price) >= 0:
        raise ValueError("volatility is outside of [sigmaLow, sigmaHigh]")

    for _ in range(max_iter):
        sigmaMid = (sigmaLow+sigmaHigh) / 2
        price_bsmMid = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                        s=s, k=k, r=r, sigma=sigmaMid, T=T, q=q)
        if(price_bsmLow - price) * (price_bsmMid - price)<0:
            sigmaHigh = sigmaMid
            price_bsmHigh = price_bsmMid
        else: 
            sigmaLow = sigmaMid
            price_bsmLow = price_bsmMid

        if abs(price_bsmMid - price) < tol:
            return sigmaMid

    print("Warning: Bisection method did not converge within max_iter")
    return None

def implied_vol_brent(option_type, exercise_style, price, s, k, r, T, q,
                       sigmaLow=1e-4, sigmaHigh=5):
    """
    Compute the implied volatility of a European call option or a European put option using the Brent's method.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        exercise_style (string): must be 'European'
        price (float): market option price
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (years)
        q (float): dividend yield
        sigmaLow (float): initial volatility lower bound
        sigmaHigh (float): initial volatility upper bound

    Returns:
        float: Implied volatility
    """

    if exercise_style != 'European':
        raise ValueError("Brent method implemented only for European options.")

    sigma = optimize.brentq(
        lambda: sigma: price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                        s=s, k=k, r=r, sigma=sigma, T=T, q=q) - price, 
        sigmaLow, sigmaHigh
    )
    return sigma
