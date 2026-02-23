# brain.py
import numpy as np
from scipy.stats import norm

def calculate_fair_value(current_price, strike_price, time_to_expiry_days, volatility=0.50):
    """
    Calculates the mathematical probability of finishing above the strike price.
    Uses a standard Normal Distribution (CDF).
    """
    # If the market has already expired
    if time_to_expiry_days <= 0:
        return 1.0 if current_price > strike_price else 0.0
    
    # Calculate Standard Deviation adjusted for time remaining
    # (volatility * sqrt of time in years)
    stdev = volatility * np.sqrt(time_to_expiry_days / 365)
    
    # Calculate how many standard deviations we are from the strike (d2)
    d2 = (np.log(current_price / strike_price) - (0.5 * stdev**2)) / stdev
    
    # norm.cdf converts d2 into a probability percentage
    return float(norm.cdf(d2))

def calculate_ev(market_price, fair_value):
    """Calculates the 'Edge' or Expected Value."""
    if market_price <= 0: return 0
    return (fair_value - market_price) / market_price