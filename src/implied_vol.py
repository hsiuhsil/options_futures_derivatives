import numpy as np
from scipy import optimize

from black_scholes_merton import price_option_bsm
from binomial_model import price_option_tree
from greeks import vega_bsm


def implied_vol_newton(option_type, exercise_style, price, s, k, r, T, q,
                       sigma0=0.1, tol=1e-8, max_iter=500, return_history=False):
    """
    Compute the implied volatility of a European call option or a European put option 
    using the Newton-Raphson method.
    
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
        return_history (bool): If True, return the sequence of volatility estimates during iteration.

    Returns:
        float or tuple: 
            If return_history is False, returns the implied volatility.
            If return_history is True, returns (implied volatility, sigma_history).
    """

    if exercise_style != 'European':
        raise ValueError("Newton method implemented only for European options.")

    sigma = sigma0
    sigma_history = [sigma] if return_history else None

    for _ in range(max_iter):

        price_bsm = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                     s=s, k=k, r=r, sigma=sigma, T=T, q=q)
        vega = vega_bsm(s, k, r, sigma, T, q)
        diff = price_bsm - price

        # check the convergence
        if abs(diff)<tol:
            return [sigma, sigma_history] if return_history else sigma

        # prevent division by very small vol
        if abs(vega)<1e-12:
            print("Warning: Vega too small, Newton-Raphson may not converge.")
            break
        sigma -= diff/vega

        # prevent negative volatility
        if sigma<=0:
            sigma = 1e-4

        if return_history:
            sigma_history.append(sigma)

    print("Newton-Raphson did not converge, try Bisection")
    sigma_bisect =  implied_vol_bisect(option_type, exercise_style, price, s, k, r, T, q,
                                       sigma_low=1e-4, sigma_high=5, tol=1e-8, max_iter=500)

    return (sigma_bisect, sigma_history) if return_history else sigma_bisect

def implied_vol_bisect(option_type, exercise_style, price, s, k, r, T, q,
                       sigma_low=1e-4, sigma_high=5, tol=1e-8, max_iter=500, return_history=False):
    """
    Compute the implied volatility of a European call option or a European put option 
    using the Bisection method.

    Parameters:
        option_type (string): either 'call' or 'put'
        exercise_style (string): must be 'European'
        price (float): market option price
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (years)
        q (float): dividend yield
        sigma_low (float): initial volatility lower bound
        sigma_high (float): initial volatility upper bound
        tol (float): tolerance for price difference
        max_iter (int): maximum number of iterations
        return_history (bool): If True, return the sequence of volatility estimates during iteration.

    Returns:
        float or tuple: 
            If return_history is False, returns the implied volatility.
            If return_history is True, returns (implied volatility, sigma_history).
    """

    if exercise_style != 'European':
        raise ValueError("Bisection method implemented only for European options.")


    price_bsm_low = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                        s=s, k=k, r=r, sigma=sigma_low, T=T, q=q)
    price_bsm_high = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
	                                s=s, k=k, r=r, sigma=sigma_high, T=T, q=q)
    f_low = price_bsm_low - price
    f_high = price_bsm_high - price
    if f_low * f_high >= 0:
        raise ValueError("volatility is outside of [sigma_low, sigma_high]")

    sigma_history = [] if return_history else None

    for _ in range(max_iter):
        sigma_mid = (sigma_low+sigma_high) / 2
        if return_history: 
            sigma_history.append(sigma_mid)

        price_bsm_mid = price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                        s=s, k=k, r=r, sigma=sigma_mid, T=T, q=q)
        f_mid = price_bsm_mid - price
        if f_low * f_mid < 0:
            sigma_high = sigma_mid
            f_high = f_mid
        else:
            sigma_low = sigma_mid
            f_low = f_mid

        if abs(price_bsm_mid - price) < tol:
            return (sigma_mid, sigma_history) if return_history else sigma_mid

    raise RuntimeError("Bisection method did not converge within max_iter")

def implied_vol_brent(option_type, exercise_style, price, s, k, r, T, q,
                       sigma_low=1e-4, sigma_high=5, N=200):
    """
    Compute the implied volatility of a European option (BSM model) 
    or a American option (Binomial tree model) using the Brent's method.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        exercise_style (string): 'European' or 'American'
        price (float): market option price
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (years)
        q (float): dividend yield
        sigma_low (float): initial volatility lower bound
        sigma_high (float): initial volatility upper bound
        N (int, option): number of steps in the binomial tree (for American options). Default is 200.

    Returns:
        float: Implied volatility
    """

    if exercise_style == 'European':
        def objective(sigma):
            return price_option_bsm(option_type=option_type, exercise_style=exercise_style,
                                    s=s, k=k, r=r, sigma=sigma, T=T, q=q) - price
    elif exercise_style == 'American':
        def objective(sigma):
            try:
                return price_option_tree(option_type=option_type, exercise_style=exercise_style,
                                         s=s, k=k, r=r, sigma=sigma, T=T, N=N, q=q) - price
            except ValueError:
                return np.nan
    else:
        raise ValueError("exercise_style must be 'European' or 'American'.")

    # check that the root is bracketed
#    if exercise_style == 'American':
#        f_low = objective(sigma_low)
#        f_high = objective(sigma_high)
#        assert f_low * f_high < 0, (
#            f"Brent method cannot run: f(sigma_low) * f(sigma_high) >= 0. "
#            f"f_low={f_low}, f_high={f_high}"
#        )

    sigma = optimize.brentq(objective, sigma_low, sigma_high)
    return sigma
