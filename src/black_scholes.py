import numpy as np
from scipy.stats import norm

class BlackScholesEngine:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        '''
        S: Current underlying asset price
        K: Strike price
        T: Time to maturity (years)
        r: Risk free interest rate (annualized)
        sigma: Volatility (annualized)
        '''
        