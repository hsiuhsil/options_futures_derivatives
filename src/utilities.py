import numpy as np

def option_payoff(option_type, s_final, k):

    if option_type == "call":
        return np.maximum(s_final - k, 0)

    elif option_type == "put":
        return np.maximum(k - s_final, 0)

    else:
        raise ValueError("option_type must be 'call' or 'put'")
