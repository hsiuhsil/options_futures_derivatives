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
        dt (float): Time between coupon payments in years (default 1).
        compounding (str, optional): "continuous" or "discrete".
        m (int, optional): Number of compounding periods per year (used for discrete compounding)
    
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
        m (int, optional): Number of compounding periods per year (used for discrete compounding)
    
    Returns
        float: Present value of the zero-coupon bond.
    """
    return face_value * discount_factor(r, T, compounding=compounding, m=m)
