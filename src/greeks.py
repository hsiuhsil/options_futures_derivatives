import numpy as np
from scipy.stats import norm
from black_scholes_merton import cumulative_prob, calculate_d1_d2

def delta(option_type, s, k, r, sigma, T):

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)

    if option_type=='call':
        return cumulative_prob(d1)
    elif option_type=='put':
        return -cumulative_prob(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def theta(option_type, s, k, r, sigma, T):

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)
    
    if option_type=='call':
        return -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*k*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*k*np.exp(-r*T)*cumulative_prob(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(s, k, r, sigma, T):
    """
    Compute the gamma of a European option (call or put).
    
    Parameters:
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)

    Returns:
        float: Option gamma
    """
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q=0)
    return norm.pdf(d1) / (s*sigma*np.sqrt(T))

def vega(s, k, r, sigma, T):
    """
    Compute the vega of a European option (call or put).
    
    Parameters:
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
 
    Returns:
        float: Option vega (per 1% change in volatility)
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q=0)
    return s*np.sqrt(T)*norm.pdf(d1)/100

def rho(option_type, s, k, r, sigma, T):
    """
    Compute the rho of a European call option or a European put option.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
 
    Returns:
        float: Option rho
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    _, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)
   
    if option_type=='call':
        return k*T*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -k*T*np.exp(-r*T)*cumulative_prob(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
