import numpy as np

def simulate_terminal_prices_mc(s, r, sigma, T, q, n_paths=1000, antithetic=False, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if antithetic:
        half = (n_paths+1)//2
        z1 = np.random.normal(0, 1, half)
        z = np.concatenate([z1, -z1])[:n_paths]
    else:
        z = np.random.normal(0, 1, n_paths)
    s1 = s*np.exp((r-q-0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    return s1

def price_option_mc(option_type, exercise_style, s, k, r, sigma, T, q, n_paths=10000, antithetic=False, control_variate=False, seed=None):

    s_final = simulate_terminal_prices_mc(s, r, sigma, T, q, n_paths, antithetic, seed=seed)
    payoffs = option_payoff(option_type, s_final, k)
    discounted_payoffs = np.exp(-r*T)*payoffs
    
    if control_variate:
        
        # control variable
        discounted_control = np.exp(-r*T) * s_final
        
        # known expectation
        control_mean = s*np.exp(-q*T)
        cov = np.cov(payoffs, discounted_control)[0,1]
        var_control = np.var(discounted_control)
        beta = cov / var_control

        adjusted = discounted_payoffs + beta*(control_mean - discounted_control)
        price = np.mean(adjusted)
        std_err = np.std(adjusted) / np.sqrt(n_paths)

    else:
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs)/np.sqrt(n_paths)

    return price, std_err
