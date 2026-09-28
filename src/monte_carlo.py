import numpy as np
from typing import Tuple

class MonteCarloEngine:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def price(self, n_sims: int = 100000, option_type: str = "call", antithetic: bool = True, seed: int = 42) -> Tuple[float, float]:
        """
        Returns:
            (estimated_price, standard_error)
        """
        np.random.seed(seed)
        effective_sims = n_sims // 2 if antithetic else n_sims

        Z = np.random.standard_normal(effective_sims)
        if antithetic:
            Z = np.concatenate([Z, -Z])

        # Step 1: Calculate the average growth trend over time T
        drift = (self.r - 0.5 * self.sigma**2) * self.T

        # Step 2: Add random daily noise scaled by volatility
        diffusion = self.sigma * np.sqrt(self.T) * Z

        # Step 3: Get all simulated final stock prices at expiration
        S_T = self.S * np.exp(drift + diffusion)

        # Work out the payout at expiration for each simulated price
        is_call = option_type.lower() == "call"
        payoffs = np.maximum(S_T - self.K, 0.0) if is_call else np.maximum(self.K - S_T, 0.0)

        # Convert future dollars back into today's money using compound interest
        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = float(np.mean(discounted_payoffs))

        # Standard error measures how noisy our simulation estimate is
        standard_error = float(np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs)))
        return price, standard_error