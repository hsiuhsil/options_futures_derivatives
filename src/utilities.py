import numpy as np

def option_payoff(option_type, s_final, k):
    """
    Compute the payoff of a European option at maturity.

    Parameters:
        option_type (str): 'call' or 'put'
        s_final (array or float): terminal stock price(s)
        k (float): strike price

    Returns:
        array or float: option payoff value(s)
    """
    if option_type == "call":
        return np.maximum(s_final - k, 0)

    elif option_type == "put":
        return np.maximum(k - s_final, 0)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

def discount_factor(r, dt, compounding="continuous", m=1):
    """
    Compute the discount factor using continuous compounding.

    Parameters
        r (float): Interest rate (annualized)
        dt (float or array-like): Time to maturity (in years)
        compounding (str, optional): "continuous" or "discrete"
        m (int, optional): Number of compounding periods per year (used for discrete compounding)

    Returns
        float or ndarray: Discount factor(s)
    """
    if compounding=="continuous":
        return np.exp(-r * dt) 
    elif compounding=="discrete":
        return (1+r/m)**(-m*dt)
    else:
        raise ValueError("compounding must be 'continuous' or 'discrete'")
