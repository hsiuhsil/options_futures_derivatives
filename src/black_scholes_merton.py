import numpy as np
from scipy.stats import norm

def cumulative_prob(x):
    """
    Calculate the cumulative distribution function (CDF) of the standard normal distribution.

    Parameters:
        x (float or array): Value(s) at which to evaluate the CDF. 

    Returns:
       float or array: CDF value(s) of the standard normal distribution at x.

    """
    return norm.cdf(x)

def calculate_d1_d2(s, k, r, sigma, T, q=0):
    """
    Vectorized calculation of d1 and d2 for Black-Scholes with dividend yield.
    
    Parameters:
        s (float or array): stock price
        k (float or array): strike price
        r (float or array): risk-free rate
        sigma (float or array): volatility
        T (float or array): time to maturity
        q (float or array): dividend yield

    Returns:
        tuple of np.ndarray: d1 and d2 terms used in the Black-Scholes-Merton formula.
    """
    s = np.asarray(s)
    k = np.asarray(k)
    r = np.asarray(r)
    sigma = np.asarray(sigma)
    T = np.asarray(T)
    q = np.asarray(q)

    if np.any(s <= 0) or np.any(k <= 0):
        raise ValueError("Stock price and strike must be positive.")
    if np.any(sigma <= 0) or np.any(T <= 0):
        raise ValueError("Volatility and maturity must be positive.")

    d1 = (np.log(s/k) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def price_option_bsm(option_type, exercise_style, s, k, r, sigma, T, q):
    """
    Compute the Black-Scholes-Merton price of a European option with dividend yield.   
 
    Parameters:
        option_type (str): either 'call' or 'put'
        exercise_style: Only 'European' is supported.
        s (float or array): stock price
        k (float or array): strike price
        r (float or array): risk-free rate
        sigma (float or array): volatility
        T (float or array): time to maturity
        q (float or array): dividend yield
    
    Returns
        float or array: Option price computed using the Black-Scholes-Merton formula.
    """
    if exercise_style!='European':
        raise ValueError("BSM applies only to European options")

    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q)

    if option_type=='call':
        return s*np.exp(-q*T)*cumulative_prob(d1) - k*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -s*np.exp(-q*T)*cumulative_prob(-d1) + k*np.exp(-r*T)*cumulative_prob(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
