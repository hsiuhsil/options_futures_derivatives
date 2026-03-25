import numpy as np
from utilities import present_value

def price_coupon_bond(face_value, coupon_rate, periods, r, dt=1, compounding="continuous", m=1):
    """
    Compute the present value (price) of a fixed-rate coupon bond.
    
    Parameters
        face_value (float): Bond face value (principal) to be repaid at maturity.
        coupon_rate (float): Annual coupon rate (as decimal, e.g., 0.05 for 5%).
        periods (int): Total number of coupon payments.
        r (float): Interest rate (annualized).
        dt (float, optional): Time between coupon payments in years (default 1).
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding).
    
    Returns
        float: Present value (price) of the coupon bond.
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    if face_value < 0:
        raise ValueError("face_value must be non-negative")

    cashflows = np.full(periods, face_value * coupon_rate)
    cashflows[-1] += face_value  # add principal to last payment
    
    times = np.arange(1, periods + 1) * dt
    
    price = present_value(cashflows, r, times, compounding=compounding, m=m)
    
    return price

def price_zero_coupon_bond(face_value, r, T, compounding="continuous", m=1):
    """
    Compute the present value (price) of a zero-coupon bond.
    
    Parameters
        face_value (float): Bond face value (principal) to be repaid at maturity.
        r (float): Interest rate (annualized).
        T (float): Time to maturity in year.
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding).
    
    Returns
        float: Present value of the zero-coupon bond.
    """
    return face_value * discount_factor(r, T, compounding=compounding, m=m)

def macaulay_duration(cashflows, r, times, compounding="continuous", m=1):
    """
    Compute the Macaulay duration of a stream of cashflows.

    Parameters
        cashflows (float or array-like): Cashflow amounts at each time in `times`.
        r (float): Annualized interest rate.
        times (float or array-like): Times to each cashflow in years. Must be the same length as `cashflows`.
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding).
 
    Returns
        float: Macaulay duration in years.
    """
    cashflows = np.array(cashflows)
    times = np.array(times)

    if len(cashflows) != len(times):
        raise ValueError("cashflows and times must have the same length")

    discount = discount_factor(r, times, compounding=compunding, m=m)
    price = np.sum(cashflows * discount)
    duration = np.sum(times * cashflows * discount) / price
    
    return duration


def convexity_coupon_bond(face_value, coupon_rate, periods, r, dt=1, delta_r=1e-4, compounding="continuous", m=1):
    """
    Compute the convexity of a coupon bond using finite differences.
    
    Parameters
        face_value (float): Bond face value (principal) to be repaid at maturity.
        coupon_rate (float): Annual coupon rate (as decimal, e.g., 0.05 for 5%).
        periods (int): Total number of coupon payments.
        r (float): Interest rate (annualized).
        dt (float, optional): Time between coupon payments in years (default 1).
        delta_r (float, optional): Small change in interest rate (default 1e-4).
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding).
    
    Returns
        float: Convexity of the coupon bond.
    """

    value_origin = price_coupon_bond(face_value, coupon_rate, periods, r, dt, compounding, m)
    value_up = price_coupon_bond(face_value, coupon_rate, periods, r + delta_r, dt, compounding, m)
    value_down = price_coupon_bond(face_value, coupon_rate, periods, r - delta_r, dt, compounding, m)

    convexity = (value_up + value_down - 2 * value_origin) / (value_origin * delta_r**2)

    return convexity


def price_change_duration_convexity(face_value, coupon_rate, periods, r, duration, convexity, delta_r, dt=1, compounding="continuous", m=1):
    """
    Compute the approximate price change of a coupon bond using duration and convexity.
    
    Parameters
        face_value (float): Bond face value (principal) to be repaid at maturity.
        coupon_rate (float): Annual coupon rate (as decimal, e.g., 0.05 for 5%).
        periods (int): Total number of coupon payments.
        r (float): Interest rate (annualized).
        duration (float): Macaulay duration.
        convexity (float): Convexity.
        delta_r (float): Change in interest rate.
        dt (float, optional): Time between coupon payments in years (default 1).
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding).
    
    Returns
        float: The approximate price change of the coupon bond.
    """

    value = price_coupon_bond(face_value, coupon_rate, periods, r, dt, compounding, m)

    price_change = value * (-duration * delta_r + 0.5 * convexity * delta_r**2)

    return price_change
