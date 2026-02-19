import numpy as np
from scipy.stats import norm
from black_scholes_merton import cumulative_prob, calculate_d1_d2

def delta(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the delta of a European call option or a European put option.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
        q (float): dividend yield
 
    Returns:
        float: Option delta
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)

    if option_type=='call':
        return np.exp(-q*T)*cumulative_prob(d1)
    elif option_type=='put':
        return -np.exp(-q*T)*cumulative_prob(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def theta(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the theta of a European call option or a European put option.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
        q (float): dividend yield
 
    Returns:
        float: Option theta
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q)
    
    if option_type=='call':
        return (
            - s*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
            + q*s*cumulative_prob(d1)*np.exp(-q*T) 
            - r*k*np.exp(-r*T)*cumulative_prob(d2)
        )
    elif option_type=='put':
        return (
            - s*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
            - q*s*cumulative_prob(-d1)*np.exp(-q*T) 
            + r*k*np.exp(-r*T)*cumulative_prob(-d2)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(s, k, r, sigma, T, q=0):
    """
    Compute the gamma of a European option (call or put).
    
    Parameters:
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
        q (float): dividend yield

    Returns:
        float: Option gamma
    """
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)
    return norm.pdf(d1)*np.exp(-q*T) / (s*sigma*np.sqrt(T))

def vega(s, k, r, sigma, T, q=0):
    """
    Compute the vega of a European option (call or put).
    
    Parameters:
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
        q (float): dividend yield 

    Returns:
        float: Option vega (per 1% change in volatility)
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)
    return s*np.sqrt(T)*norm.pdf(d1)*np.exp(-q*T)/100

def rho(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the rho of a European call option or a European put option.
    
    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        T (float): Time to maturity (years)
        q (float): dividend yield 

    Returns:
        float: Option rho
    """

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    _, d2 = calculate_d1_d2(s, k, r, sigma, T, q)
   
    if option_type=='call':
        return k*T*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -k*T*np.exp(-r*T)*cumulative_prob(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
