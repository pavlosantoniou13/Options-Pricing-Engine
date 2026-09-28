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

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def _d1_d2(self):
        '''
        Helper method to calculate d1 and d2 statistical terms
        d2 = probability of exercising in the money
        d1 = weights that probability by the expected asset payoff
        '''
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self,option_type: str = "call") -> float:
        # Calculate the theoretical fair value of a call or put option
        d1, d2 = self._d1_d2()
        if option_type.lower() == "call":
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) *norm.cdf(d2)
        elif option_type.lower == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        raise ValueError("Option_type must be a 'call' or 'put'")

    # The Greeks
    def delta(self, option_type: str = 'call') -> float:
        '''
        Rate of change of option price with respect to stock price
        call range 0 to 1
        put range -1 to 0
        '''
        d1, _ = self._d1_d2()
        return norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(d1) - 1.0

    def gamma(self) -> float:
        '''
        Rate of change of Delta with respect to stock price.
        Measures the curvature / acceleration which is identical for both calls and puts.
        '''
        d1, _ = self._d1_d2()
        return norm.cdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        '''
        Sensitivity to changes in market volatility (sigma)
        Higher volatility increases the value of both calls and puts equally.   
        '''
        d1, _ = self._d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self, option_type: str = 'call') -> float:
        '''
        Time decay = how much value the option loses at time passes.
        Usually negative because holding an option becomes less valuable as expirty nears.
        '''
        d1, d2 = self._d1_d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type.lower() == "call":
            return term1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return term1 + self.r * self.K * np.exp(-self.r *  self.T) * norm.cdf(-d2)

    def rho(self, option_type: str = "call") -> float:
        '''
        Sensitivity to interest rates (r).
        calls benefit from higher rates (you pay cash later).
        puts lose value when rates rise (you receive cash later).
        '''
        _, d2 = self._d1_d2()
        if option_type.lower() == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)