import pytest
from src.black_scholes import BlackScholesEngine
from src.binomial_tree import BinomialTreeEngine
from src.monte_carlo import MonteCarloEngine

def test_model_convergence():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    
    # 1. Closed-form ground truth
    bs = BlackScholesEngine(S, K, T, r, sigma)
    bs_call = bs.price("call")
    
    # 2. Binomial tree convergence (N=500)
    tree = BinomialTreeEngine(S, K, T, r, sigma, N=500)
    tree_call = tree.price("call", exercise="european")
    assert abs(tree_call - bs_call) < 0.02, f"Binomial Tree error too large: {tree_call} vs {bs_call}"

    # 3. Monte Carlo convergence (within 99% CI: 2.58 * SE)
    mc = MonteCarloEngine(S, K, T, r, sigma)
    mc_call, mc_se = mc.price(n_sims=200000, option_type="call")
    assert abs(mc_call - bs_call) < 2.58 * mc_se, f"MC outside 99% CI: {mc_call} vs {bs_call}"

def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = BlackScholesEngine(S, K, T, r, sigma)
    call = bs.price("call")
    put = bs.price("put")
    parity_lhs = call - put
    parity_rhs = S - K * (2.718281828459045 ** (-r * T))
    assert pytest.approx(parity_lhs, 1e-4) == parity_rhs