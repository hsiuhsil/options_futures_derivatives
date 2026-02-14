# calculate and plot the binomial trees

import numpy as np

def _binomial_engine(s, k, u, d, T, N, p, r, dt, 
                     is_call, is_american):

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

def option_price_no_volatility(option_type, exercise_style, s, k, up, down, dt_months, T, r, return_tree=False):
    # from ch 13.4
    # option_type: 'call' or 'put'
    # exercise_style: 'European' or 'American'
    # s: the current stock price
    # k: the strike price
    # up: the amount that the stock price will move up in percentage
    # down: the amount that the stock price will move down in percentage
    # dt_months: the duration of each step (in months)
    # T: the duration of all steps (in years)
    # r: the risk-free rate 

    if option_type not in ('call','put'):
        raise ValueError("option_type is either `call` or `put`")
    
    isCall, isPut = (True, False) if option_type == 'call' else (False, True)
    isAmerican, isEuropean = (True, False) if exercise_style == 'American' else (False, True)

    N = int(T/(dt_months/12))
    dt = T/N

    if abs(N * dt - T) > 1e-10:
        raise ValueError("Inconsistent step size.")
    
    u = 1+up*0.01
    d = 1-down*0.01
    p = (np.exp(r*dt)-d)/(u-d)
    if not (0 < p < 1):
        raise ValueError("Arbitrage detected.")

    price, stocks, options, timeline = _binomial_engine(s, k, u, d, T, N, p, r, dt,
                                                        is_call, is_american)
    
    if return_tree:
        return price, stocks, options, timeline
    else:
        return price

def option_price(option_type, exercise_style, s, k, vol, dt_months, T, r):
    # from ch 13.7
    # option_type: 'call' or 'put'
    # exercise_style: 'European' or 'American'
    # s: the current stock price
    # k: the strike price
    # vol: the volatility (0<vol<1)
    # dt_months: the duration of each step (in months)
    # T: the duration of all steps (in years)
    # r: the risk-free rate 

    if option_type not in ('call','put'):
        raise ValueError("option_type is either `call` or `put`")
    
    isCall, isPut = (True, False) if option_type == 'call' else (False, True)
    isAmerican, isEuropean = (True, False) if exercise_style == 'American' else (False, True)

    N = int(T/(dt_months/12))
    dt = T/N
    if abs(N * dt - T) > 1e-10:
        raise ValueError("Inconsistent step size.")
    
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    if not (0 < p < 1):
        raise ValueError("Arbitrage detected.")

    price, stocks, options, timeline = _binomial_engine(s, k, u, d, T, N, p, r, dt,
                                                        is_call, is_american) 
    if return_tree:
        return price, stocks, options, timeline
    else:
        return price
