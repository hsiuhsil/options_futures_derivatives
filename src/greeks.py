import numpy as np
from scipy.stats import norm
from black_scholes_merton import cumulative_prob, calculate_d1_d2
from binomial_model import price_option_tree, price_option_tree_no_volatility

def delta_bsm(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the delta of a European call option or a European put option using the Black-Scholes-Merton model.
    
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

    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)

    if option_type=='call':
        return np.exp(-q*T)*cumulative_prob(d1)
    elif option_type=='put':
        return -np.exp(-q*T)*cumulative_prob(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def theta_bsm(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the theta of a European call option or a European put option using the Black-Scholes-Merton model.
    
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

    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
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

def gamma_bsm(s, k, r, sigma, T, q=0):
    """
    Compute the gamma of a European option (call or put) using the Black-Scholes-Merton model.
    
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
    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)
    return norm.pdf(d1)*np.exp(-q*T) / (s*sigma*np.sqrt(T))

def vega_bsm(s, k, r, sigma, T, q=0):
    """
    Compute the vega of a European option (call or put) using the Black-Scholes-Merton model.
    
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

    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")
    d1, _ = calculate_d1_d2(s, k, r, sigma, T, q)
    return s*np.sqrt(T)*norm.pdf(d1)*np.exp(-q*T)/100

def rho_bsm(option_type, s, k, r, sigma, T, q=0):
    """
    Compute the rho of a European call option or a European put option using the Black-Scholes-Merton model.
    
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

    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")
    _, d2 = calculate_d1_d2(s, k, r, sigma, T, q)
   
    if option_type=='call':
        return k*T*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -k*T*np.exp(-r*T)*cumulative_prob(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def delta_tree(option_type, exercise_style, s, k, sigma, T, N, r, q=0):
    """
    Compute the delta of a European call option or a European put option using the binomial tree  model.

    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        sigma (float): Volatility
        T (float): Time to maturity (years)
        N (int): Number of steps
        r (float): Risk-free rate
        q (float): dividend yield
 
    Returns:
        float: Option delta
    """

    if N <= 0:
        raise ValueError("Number of steps N must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")

    if option_type not in ['call','put']:
        raise ValueError("option_type must be 'call' or 'put'")

    price, stocks, options, timeline = price_option_tree(option_type, exercise_style, s, k, sigma, T, N, r, q,  return_tree=True)
    return (options[1][0]-options[1][1]) / (stocks[1][0]-stocks[1][1])  

def theta_tree(option_type, exercise_style, s, k, sigma, T, N, r, q=0):
    """
    Compute the theta of a European or American option using the binomial tree model.

    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        sigma (float): Volatility
        T (float): Time to maturity (years)
        N (int): Number of steps
        r (float): Risk-free rate
        q (float): dividend yield

    Returns:
        float: Option theta (per year)
    """

    if N < 2:
        raise ValueError("Number of steps N must be >= 2.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")

    price, stocks, options, timeline = price_option_tree(option_type, exercise_style, s, k, sigma, T, N, r, q, return_tree=True)

    dt = T / N

    V0 = options[0][0]
    V_ud = options[2][1]

    theta = (V_ud - V0) / (2 * dt)

    return theta
   
def gamma_tree(option_type, exercise_style, s, k, sigma, T, N, r, q=0):
    """
    Compute the gamma of a European call option or a European put option using the binomial tree model.

    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        sigma (float): Volatility
        T (float): Time to maturity (years)
        N (int): Number of steps
        r (float): Risk-free rate
        q (float): dividend yield

    Returns:
        float: Option gamma
    """

    if N <= 1:
        raise ValueError("Number of steps N must be >= 2.")
    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")

    if option_type not in ['call','put']:
        raise ValueError("option_type must be 'call' or 'put'")

    price, stocks, options, timeline = price_option_tree(option_type, exercise_style, s, k, sigma, T, N, r, q,  return_tree=True)
    delta_up = (options[2][0] - options[2][1]) / (stocks[2][0] - stocks[2][1])
    delta_down = (options[2][1] - options[2][2]) / (stocks[2][1] - stocks[2][2])

    return (delta_up - delta_down) / (stocks[1][0] - stocks[1][1])

 
def vega_tree(option_type, exercise_style, s, k, sigma, T, N, r, q=0, epsilon=1e-4):
    """
    Compute the vega of a European option (call or put) using the binomial tree model. 

    Parameters:
        option_type (string): either 'call' or 'put'
        s (float): Current stock price
        k (float): Strike price
        sigma (float): Volatility
        T (float): Time to maturity (years)
        N (int): Number of steps
        r (float): Risk-free rate
        q (float): dividend yield
        epsilon (float): a tiny amount of perturbation

    Returns:
        float: Option vega (per 1% change in volatility)
    """

    if np.any(sigma <= 0):
        raise ValueError("Volatility must be positive.")
    if np.any(T <= 0):
        raise ValueError("Maturity must be positive.")
    if np.any(epsilon <= 0):
        raise ValueError("epsilon must be positive.")
    if np.any(sigma-epsilon <0):
        raise ValueError("epsilon too large relative to sigma")
    
    price1 = price_option_tree(option_type, exercise_style, s, k, sigma+epsilon, T, N, r, q, return_tree=False)
    price2 = price_option_tree(option_type, exercise_style, s, k, sigma-epsilon, T, N, r, q, return_tree=False)
    vega = (price1 - price2)/(2*epsilon)
    return vega / 100
