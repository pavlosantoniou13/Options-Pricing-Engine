import numpy as np

class BinomialTreeEngine:
    def __init__(self, S: float, K: float, T:float, r: float, sigma: float, N: int = 500):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.N = int(N)

    def price(self, option_type: str = "call", exercise: str = "european") -> float:
        '''
        Price European or American options via backward induction on a binomial tree.
        European: can only be exercised at the expiration date (maturity).
        American: can be exercised early at any node before maturity.
        '''
        # Step size: length of each time step in years
        dt = self.T / self.N

        # CRR up/down price movement multipliers per time step
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u

        # Risk-neutral probability of an upward move
        p = (np.exp(self.r * dt) - d) / (u - d)

        # Discount factor to bring cash flow back by one step dt
        discount = np.exp(-self.r * dt)

        # Calculate all possible stock prices at final expiration step N
        # Node j corresponds to j 'up' moves and (N - j) 'down' moves
        j = np.arrange(self.N + 1)
        S_T = self.s * (u ** j) * (d ** (self.N - j))

        # Intrinsic option payoff at expiration: max(S - K, 0) for call, max(K - S, 0) for put
        is_call = option_type.lower() == "call"
        values = np.maximum(S_T - self.K, 0.0) if is_call else np.maximum(self.K - S_T, 0.0)
        # Backward induction: step backwards from step N-1 down to step 0 (today)
        is_american = exercise.lower() == "american"
        for i in range(self.N - 1, -1, -1):
            # Continuation value: expected discounted value from holding the option to next step
            values = discount * (p * values[1:] + (1.0 -p) * values[:-1])
            # Check for optimal early exercise (American options only)s
            if is_american:
                j_i = np.arrange(i + 1)
                S_node = self.S * (u ** j_i) * (d ** (i - j_i))
                early_exercise = np.maximum(S_node - self.K, 0.0) if is_call else np.maximum(self.K - S_node, 0.0)
                # Option value is the maximum of holding vs exercising immediately
                values = np.maximum(values, early_exercise)
        # Root node of the tree represents the fair option price today
        return float(values[0])
