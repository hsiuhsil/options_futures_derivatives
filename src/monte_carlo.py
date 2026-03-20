import numpy as np
from utilities import option_payoff
from black_scholes_merton import price_option_bsm

def simulate_terminal_prices_mc(s, r, sigma, T, q, n_paths=1000, antithetic=False, seed=None):
    """
    Simulate terminal stock prices using Monte Carlo under the risk-neutral measure.

    Parameters:
        s (float): current stock price
        r (float): risk-free rate
        sigma (float): volatility of the underlying asset
        T (float): time to maturity (years)
        q (float): dividend yield
        n_paths (int): number of Monte Carlo simulation paths
        antithetic (bool): if True, use antithetic variates for variance reduction
        seed (int): random seed for reproducibility

    Returns:
        array: simulated terminal stock prices
    """
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
    """
    Price an option using Monte Carlo simulation.

    Parameters:
        option_type (str): 'call' or 'put'
        exercise_style (str): 'European' only
        s (float): current stock price
        k (float): strike price
        r (float): risk-free rate
        sigma (float): volatility of the underlying asset
        T (float): time to maturity (years)
        q (float): dividend yield
        n_paths (int): number of Monte Carlo simulation paths
        antithetic (bool): if True, use antithetic variates
        control_variate (bool): if True, apply control variate variance reduction
        seed (int): random seed for reproducibility

    Returns:
        price (float): estimated option price
        std_err (float): Monte Carlo standard error of the estimate
    """
    s_final = simulate_terminal_prices_mc(s, r, sigma, T, q, n_paths, antithetic, seed=seed)
    payoffs = option_payoff(option_type, s_final, k)
    discounted_payoffs = np.exp(-r*T)*payoffs
    
    if control_variate:
        
        # control variable
        discounted_control = np.exp(-r*T) * s_final
        
        # known expectation
        control_mean = s*np.exp(-q*T)
        cov = np.cov(payoffs, discounted_control)[0,1]
        var_control = np.var(discounted_control, ddof=1)
        beta = cov / var_control
        
        adjusted = discounted_payoffs + beta*(control_mean - discounted_control)
        price = np.mean(adjusted)
        std_err = np.std(adjusted, ddof=1) / np.sqrt(n_paths)

    else:
        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1)/np.sqrt(n_paths)

    return price, std_err


def price_american_lsmc(option_type, s, k, r, sigma, T, q, n_steps, n_paths,
                        antithetic=False, control_variate=False, seed=None,
                        return_exercise=False,
                        poly_order=2,
                        regression="ols",
                        ridge_alpha=1e-6):
    """
    Price an American option using Longstaff-Schwartz Monte Carlo (LSMC).

    Parameters
        option_type (str): 'call' or 'put'
        s (float): initial stock price
        k (float): strike price
        r (float): risk-free rate
        sigma (float): volatility
        T (float): time to maturity (in years)
        q (float): dividend yield
        n_steps (int): number of time steps
        n_paths (int): number of simulated paths

        antithetic (bool): use antithetic variates
        control_variate (bool): use European option as control variate
        seed (int or None): random seed
        return_exercise (bool): return exercise matrix and paths

        poly_order (int): polynomial order for regression (default=2)
        regression (str): 'ols' or 'ridge'
        ridge_alpha (float): ridge regularization strength

    Returns
        price (float)
        std_err (float)
        exercise_matrix (optional)
        stocks (optional)
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    discount = np.exp(-r * dt)

    # simulate stock returns
    if antithetic:
        half = (n_paths + 1) // 2
        Z_half = np.random.normal(size=(half, n_steps))
        Z = np.vstack([Z_half, -Z_half])[:n_paths]
    else:
        Z = np.random.normal(size=(n_paths, n_steps))

    stocks = np.zeros((n_paths, n_steps + 1))
    stocks[:, 0] = s

    for t in range(1, n_steps + 1):
        stocks[:, t] = stocks[:, t-1] * np.exp(
            (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
        )

    # the payoff matrix
    if option_type == "call":
        payoff = np.maximum(stocks - k, 0)
    elif option_type == "put":
        payoff = np.maximum(k - stocks, 0)
    else:
        raise ValueError("option_type must be call or put")

    cashflow = payoff[:, -1]
    exercise_matrix = np.zeros((n_paths, n_steps + 1))

    # the backward induction
    for t in range(n_steps - 1, 0, -1):

        in_money = payoff[:, t] > 0
        X = stocks[in_money, t]
        Y = cashflow[in_money] * discount

        if len(X) == 0:
            cashflow *= discount
            continue

        # the polynomial basis
        A = np.vstack([X**i for i in range(poly_order + 1)]).T

        # the regression
        if regression == "ols":
            beta = np.linalg.lstsq(A, Y, rcond=None)[0]

        elif regression == "ridge":
            # beta = (A^T A + alpha I)^(-1) A^T Y
            ATA = A.T @ A
            ridge = ridge_alpha * np.eye(ATA.shape[0])
            beta = np.linalg.solve(ATA + ridge, A.T @ Y)

        else:
            raise ValueError("regression must be 'ols' or 'ridge'")

        continuation = A @ beta
        exercise = payoff[in_money, t]

        exercise_now = exercise > continuation

        idx = np.where(in_money)[0]
        exercise_matrix[idx[exercise_now], t] = 1

        cashflow[idx[exercise_now]] = exercise[exercise_now]
        cashflow[idx[~exercise_now]] *= discount

    # discount final cashflows
    values = cashflow * discount
    final_values = values

    # control variate
    if control_variate:
        if option_type == "call":
            euro_payoff = np.maximum(stocks[:, -1] - k, 0)
        else:
            euro_payoff = np.maximum(k - stocks[:, -1], 0)

        euro_discounted = np.exp(-r * T) * euro_payoff

        cov = np.cov(values, euro_discounted)[0, 1]
        var_control = np.var(euro_discounted, ddof=1)

        beta_cv = cov / var_control if var_control > 0 else 0

        price_bsm = price_option_bsm(option_type, "European", s, k, r, sigma, T, q)

        final_values = values + beta_cv * (price_bsm - euro_discounted)

    price = np.mean(final_values)
    std_err = np.std(final_values, ddof=1) / np.sqrt(n_paths)

    if return_exercise:
        return price, std_err, exercise_matrix, stocks
    else:
        return price, std_err
