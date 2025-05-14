#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Import FSRO variants
from fsro_variants.fsro_modified import FSRO as FSRO_Modified
from fsro_variants.fsro_original import FSRO as FSRO_Original

# Import other algorithms
from algorithms import de, gwo, hho, mfo, pso, sma, woa

# --------------------
# Define hardcoded benchmark functions
# --------------------
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def f3(x):
    return np.sum(np.cumsum(x)**2)

def f4(x):
    return np.max(np.abs(x))

def f5(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def f6(x):
    return np.sum(np.floor(x + 0.5)**2)

def f7(x):
    return np.sum(np.arange(1, len(x)+1) * x**4) + np.random.rand()

# --------------------
# Function dictionary
# --------------------
benchmark_functions = {
    "F1 (Sphere)": f1,
    "F2": f2,
    "F3": f3,
    "F4": f4,
    "F5": f5,
    "F6": f6,
    "F7": f7
}

# --------------------
# Configuration
# --------------------
DIM = 30
RUNS = 10
MAX_GEN = 500

# --------------------
# Main benchmarking loop
# --------------------
for func_name, func in benchmark_functions.items():
    print(f"\n\n===== Benchmark: {func_name} =====")
    results = {}
    histories = {}

    for alg_name in [
        "FSRO Modified", "FSRO Original",
        "DE", "GWO", "HHO", "MFO", "PSO", "SMA", "WOA"
    ]:
        print(f"\nRunning {alg_name}...")
        alg_results = []
        alg_histories = []

        for _ in range(RUNS):
            # ----------------------
            # Algorithm Execution Routing
            # ----------------------
            if alg_name == "FSRO Modified":
                pop_size = DIM * 2
                optimizer = FSRO_Modified(func, pop_size=pop_size, dim=DIM, max_iter=MAX_GEN)
                solution, best_fit, history = optimizer.optimize()

            if alg_name == "FSRO Original":
                optimizer = FSRO_Original(fobj=func, dim=DIM, max_iter=MAX_GEN)
                solution, best_fit, history = optimizer.optimize()


            elif alg_name == "DE":
                solution, best_fit, history = de.optimize(func, DIM, MAX_GEN)

            elif alg_name == "GWO":
                solution, best_fit, history = gwo.optimize(func, DIM, MAX_GEN)

            elif alg_name == "HHO":
                solution, best_fit, history = hho.optimize(func, DIM, MAX_GEN)

            elif alg_name == "MFO":
                solution, best_fit, history = mfo.optimize(func, DIM, MAX_GEN)

            elif alg_name == "PSO":
                solution, best_fit, history = pso.optimize(func, DIM, MAX_GEN)

            elif alg_name == "SMA":
                solution, best_fit, history = sma.optimize(func, DIM, MAX_GEN)

            elif alg_name == "WOA":
                solution, best_fit, history = woa.optimize(func, DIM, MAX_GEN)

            else:
                raise ValueError(f"Unknown algorithm: {alg_name}")

            alg_results.append(best_fit)
            alg_histories.append(history)

        results[alg_name] = np.array(alg_results)
        histories[alg_name] = np.array(alg_histories)

    # --------------------
    # Wilcoxon statistical analysis: FSRO Modified vs others
    # --------------------
    print("\n--- Wilcoxon Test: FSRO Modified vs Others ---")
    mod_scores = results["FSRO Modified"]
    for other_name in results:
        if other_name == "FSRO Modified":
            continue
        stat, p_value = wilcoxon(mod_scores, results[other_name])
        print(f"FSRO Modified vs {other_name}: W={stat:.4f}, p-value={p_value:.6f}")

    # --------------------
    # Plot convergence curves
    # --------------------
    plt.figure(figsize=(12, 8))
    for alg_name, hist in histories.items():
        mean_curve = np.mean(hist, axis=0)
        if "FSRO" in alg_name:
            plt.plot(mean_curve, label=alg_name, linewidth=2.5)
        else:
            plt.plot(mean_curve, label=alg_name, linestyle="--", linewidth=1.8)

    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Best Fitness (log scale)", fontsize=14)
    plt.title(f"Convergence Curve - {func_name}", fontsize=16)
    plt.yscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
