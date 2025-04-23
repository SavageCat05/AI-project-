import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import pso, de, gwo, mfo, woa, hho, sma
from benchmarks import cec14, cec17, cec20, cec22  

# Settings
n_runs = 50
max_evals = 60000
dim = 30  # CEC functions usually 30D
save_folder = "results"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# List of algorithms
algorithms = {
    "PSO": pso.optimize,
    "DE": de.optimize,
    "GWO": gwo.optimize,
    "MFO": mfo.optimize,
    "WOA": woa.optimize,
    "HHO": hho.optimize,
    "SMA": sma.optimize,
}

# List of benchmark functions
benchmark_sets = {
    "CEC14": cec14.get_functions(),
    "CEC17": cec17.get_functions(),
    "CEC20": cec20.get_functions(),
    "CEC22": cec22.get_functions(),
}

# Main runner
for set_name, functions in benchmark_sets.items():
    print(f"\nRunning Benchmark Set: {set_name}")
    
    for func_idx, (func_name, func) in enumerate(functions.items(), 1):
        bounds = [(-100, 100)] * dim  # CEC standard bounds (adjust if needed)
        print(f"\nFunction {func_idx}: {func_name}")

        all_results = {}

        for algo_name, optimizer in algorithms.items():
            print(f"  Running {algo_name}...")
            best_scores = []
            all_convergence = []

            for run in range(n_runs):
                best_solution, convergence_curve = optimizer(func, bounds, max_evals)
                best_score = func(best_solution)
                best_scores.append(best_score)
                all_convergence.append(convergence_curve)

            all_results[algo_name + "_mean"] = np.mean(best_scores)
            all_results[algo_name + "_std"] = np.std(best_scores)

            # Save convergence curve plot (average curve)
            mean_curve = np.mean(all_convergence, axis=0)
            plt.plot(mean_curve, label=algo_name)

            # Optionally save per-run convergence data too if needed

        plt.title(f"Convergence Curve - {func_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_folder}/{set_name}_{func_name}_convergence.png")
        plt.clf()

        # Save all_results to CSV
        df = pd.DataFrame([all_results])
        csv_file = f"{save_folder}/{set_name}_results.csv"
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)

print("\n All experiments completed!")
