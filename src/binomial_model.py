import numpy as np

def _binomial_engine(s, k, u, d, T, N, p, r, dt, 
                     isCall, isPut, isAmerican):
    """
    Core engine for binomial option pricing.

    This function constructs the stock price tree and performs
    backward induction to compute the option value.

    Parameters:
        s (float): current stock price
        k (float): strike price
        u (float): up movement multiplier
        d (float): down movement multiplier
        T (float): time to maturity (years)
        N (int): number of time steps
        p (float): risk-neutral probability of an up move
        r (float): risk-free rate
        dt (float): time step size
        isCall (bool): True if call option
        isPut (bool): True if put option
        isAmerican (bool): True if American exercise allowed

    Returns:
        price (float): option price
        stocks (list): stock price tree
        options (list): option value tree
        timeline (array): time grid
    """

    stocks, options = [], []
    timeline = np.linspace(0, T, N+1, endpoint=True)

    for n in range(N+1):
        stock, option = [], []
        for i in range(N-n+1):
            x = s*u**(N-n-i)*d**i
            if n == 0:
                if isCall: y = max(x-k,0) 
                else: y = max(k-x,0)
            else:
                y = (p*options[-1][i]+(1-p)*options[-1][i+1])*np.exp(-r*dt)
                if isAmerican: 
                    if isCall: y = max(x-k, y) 
                    if isPut: y = max(k-x, y)
            stock.append(x)
            option.append(y)
            
        stocks.append(stock)
        options.append(option)

    return options[-1][0], stocks[::-1], options[::-1], timeline

def price_option_tree_no_volatility(option_type, exercise_style, s, k, up, down, T, N, r, q=0, return_tree=False):
    """
    Price an option using a binomial tree with predefined up and down movements.

    Parameters:
        option_type (str): 'call' or 'put'
        exercise_style (str): 'European' or 'American'
        s (float): current stock price
        k (float): strike price
        up (float): percentage upward move of the stock price
        down (float): percentage downward move of the stock price
        T (float): time to maturity (years)
        N (int): number of time steps
        r (float): risk-free rate
        q (float): dividend yield (not used in this implementation)
        return_tree (bool): if True, also return the binomial trees

    Returns:
        float: option price

        If return_tree=True, also returns:
            stocks (list): stock price tree
            options (list): option value tree
            timeline (array): time grid
    """

    if option_type not in ('call','put'):
        raise ValueError("option_type is either `call` or `put`")
    if exercise_style not in ("European", "American"):
        raise ValueError("exercise_style must be 'European' or 'American'")    

    isCall, isPut = (True, False) if option_type == 'call' else (False, True)
    isAmerican, isEuropean = (True, False) if exercise_style == 'American' else (False, True)

    dt = T/N

    if abs(N * dt - T) > 1e-10:
        raise ValueError("Inconsistent step size.")
    
    u = 1+up*0.01
    d = 1-down*0.01
    p = (np.exp(r*dt)-d)/(u-d)
    if not (0 < p < 1):
        raise ValueError("Arbitrage detected.")

    price, stocks, options, timeline = _binomial_engine(s, k, u, d, T, N, p, r, dt,
                                                        isCall, isPut, isAmerican)
    
    if return_tree:
        return price, stocks, options, timeline
    else:
        return price

def price_option_tree(option_type, exercise_style, s, k, r, sigma, T, N, q=0, return_tree=False):
    """
    Price an option using a binomial tree with volatility-based movements.

    Parameters:
        option_type (str): 'call' or 'put'
        exercise_style (str): 'European' or 'American'
        s (float): current stock price
        k (float): strike price
        r (float): risk-free rate
        sigma (float): volatility of the underlying asset
        T (float): time to maturity (years)
        N (int): number of time steps
        q (float): dividend yield
        return_tree (bool): if True, also return the binomial trees

    Returns:
        float: option price

        If return_tree=True, also returns:
            stocks (list): stock price tree
            options (list): option value tree
            timeline (array): time grid
    """

    if option_type not in ('call','put'):
        raise ValueError("option_type is either `call` or `put`")

    if exercise_style not in ("European", "American"):
        raise ValueError("exercise_style must be 'European' or 'American'")   
 
    isCall, isPut = (True, False) if option_type == 'call' else (False, True)
    isAmerican, isEuropean = (True, False) if exercise_style == 'American' else (False, True)

    dt = T/N
    if abs(N * dt - T) > 1e-10:
        raise ValueError("Inconsistent step size.")
    
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt)-d)/(u-d)
    if not (0 < p < 1):
        raise ValueError("Arbitrage detected.")

    price, stocks, options, timeline = _binomial_engine(s, k, u, d, T, N, p, r, dt,
                                                        isCall, isPut, isAmerican) 
    if return_tree:
        return price, stocks, options, timeline
    else:
        return price
