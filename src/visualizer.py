import numpy as np
import matplotlib.pyplot as plt
from src.black_scholes import BlackScholesEngine

def plot_greeks(K=100.0, T=1.0, r=0.05, sigma=0.20):
    spot_range = np.linspace(50, 150, 100)
    
    deltas_call, gammas, vegas, thetas = [], [], [], []
    for s in spot_range:
        engine = BlackScholesEngine(S=s, K=K, T=T, r=r, sigma=sigma)
        deltas_call.append(engine.delta("call"))
        gammas.append(engine.gamma())
        vegas.append(engine.vega())
        thetas.append(engine.theta("call"))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(spot_range, deltas_call, color="blue")
    axs[0, 0].set_title("Delta (Call)")
    axs[0, 0].set_xlabel("Spot Price")

    axs[0, 1].plot(spot_range, gammas, color="red")
    axs[0, 1].set_title("Gamma")
    axs[0, 1].set_xlabel("Spot Price")

    axs[1, 0].plot(spot_range, vegas, color="green")
    axs[1, 0].set_title("Vega")
    axs[1, 0].set_xlabel("Spot Price")

    axs[1, 1].plot(spot_range, thetas, color="purple")
    axs[1, 1].set_title("Theta (Call)")
    axs[1, 1].set_xlabel("Spot Price")

    plt.tight_layout()
    plt.savefig("notebooks/greeks_profile.png")
    plt.show()

if __name__ == "__main__":
    plot_greeks()