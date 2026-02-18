import numpy as np
from scipy.stats import norm
from black_scholes_merton import cumulative_prob, calculate_d1_d2

def delta(option_type, s, k, r, sigma, T):

    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)

    if option_type=='call':
        return cumulative_prob(d1)
    elif option_type=='put':
        return cumulative_prob(d1) - 1

def theta(option_type, s, k, r, sigma, T):

    if T <= 0:
        raise ValueError("Maturity must be positive.")
    d1, d2 = calculate_d1_d2(s, k, r, sigma, T, q=0)
    
    if option_type=='call':
        return -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*k*np.exp(-r*T)*cumulative_prob(d2)
    elif option_type=='put':
        return -s*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*k*np.exp(-r*T)*cumulative_prob(-d2)


