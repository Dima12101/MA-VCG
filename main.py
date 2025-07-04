"""
Multi-agent edge-computing prototype with topology-aware Vickrey auctions.
Tested on Python 3.10+

pip install networkx numpy matplotlib
"""

import random
import numpy as np

from environment import EdgeComputingSystem

# --------------------------------------------------------------------------- #
#                                   DEMO                                      #
# --------------------------------------------------------------------------- #
def main():
    random.seed(42)
    np.random.seed(42)

    system = EdgeComputingSystem(n_nodes=10, n_devices=20)
    results = system.run(n_rounds=50)

    print("=== Simulation results (50 rounds) ===")
    print(f"Average social welfare   : {results['avg_welfare']:.3f}")
    print(f"Average allocation eff.  : {results['avg_efficiency']:.3f}")
    print(f"Average fairness (Jain)  : {results['avg_fairness']:.3f}")


if __name__ == "__main__":
    main()
