import numpy as np

def historical_var(returns, x=0.05):
    """
    Compute Historical Value-at-Risk (VaR).

    Parameters:
        returns (array): portfolio returns (e.g., daily log-returns)
        x (float): Significance level (e.g., 0.05 for 95% VaR)

    Returns:
        float: VaR value (positive number representing loss)
    """

    if len(returns) == 0:
        return np.nan
        
    clean_returns = returns[~np.isnan(returns)]
    
    var_threshold = np.percentile(clean_returns, 100*x)
    return -var_threshold

def historical_es(returns, x=0.05):
    """
    Compute Historical Expected Shortfall (ES).

    Parameters:
        returns (array): portfolio returns (e.g., daily log-returns)
        x (float): Significance level (e.g., 0.05 for 95% VaR)

    Returns:
        float: Expected Shortfall
    """
    clean_returns = returns[~np.isnan(returns)]

    if len(clean_returns) == 0:
        return np.nan
        
    var_threshold = historical_var(clean_returns, x)
    tail_losses = clean_returns[clean_returns <= -var_threshold]

    if len(tail_losses)==0:
        return var_threshold

    return -tail_losses.mean()

