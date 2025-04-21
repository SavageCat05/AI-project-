
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# === Placeholder: Import your FSRO algorithm ===
# from fsro_module import fsro_optimize

# === Benchmark setup ===
NUM_RUNS = 50
NUM_FUNCTIONS = 30
DIM = 30
MAX_EVALS = 60000

# Placeholder for CEC function suite
def get_cec_functions():
    return [lambda x: np.sum(x**2) for _ in range(NUM_FUNCTIONS)]  # Use real CEC loader here

# === Evaluation tracking wrapper ===
class EvaluationWrapper:
    def __init__(self, func, max_evals):
        self.func = func
        self.counter = 0
        self.max_evals = max_evals

    def evaluate(self, x):
        if self.counter >= self.max_evals:
            return float('inf')
        self.counter += 1
        return self.func(x)

# === Dummy FSRO runner (replace with your actual optimizer) ===
def fsro_optimize(obj_wrapper, dim, max_evals):
    best_fitness = float("inf")
    convergence_curve = []
    for _ in range(max_evals):
        x = np.random.uniform(-100, 100, size=dim)
        fit = obj_wrapper.evaluate(x)
        if fit < best_fitness:
            best_fitness = fit
        convergence_curve.append(best_fitness)
    return best_fitness, convergence_curve

# === Benchmarking ===
cec_functions = get_cec_functions()
results_matrix = np.zeros((NUM_RUNS, NUM_FUNCTIONS))
convergence_data = []

for f_idx, func in enumerate(cec_functions):
    print(f"Running function {f_idx + 1}/{NUM_FUNCTIONS}")
    func_convergence = []

    for run in range(NUM_RUNS):
        wrapper = EvaluationWrapper(func, MAX_EVALS)
        best, curve = fsro_optimize(wrapper, DIM, MAX_EVALS)
        results_matrix[run, f_idx] = best
        func_convergence.append(curve)

    convergence_data.append(np.mean(func_convergence, axis=0))  # Mean convergence per function

# === Analysis ===
means = np.mean(results_matrix, axis=0)
stds = np.std(results_matrix, axis=0)
ranks = pd.Series(means).rank().values

# Create result summary table
summary = pd.DataFrame({
    "Function": [f"F{i+1}" for i in range(NUM_FUNCTIONS)],
    "Mean": means,
    "StdDev": stds,
    "Rank": ranks
})
print(summary)

# === Save results ===
summary.to_csv("fsro_summary_results.csv", index=False)
np.save("fsro_all_convergence.npy", np.array(convergence_data))

# === Plot sample convergence curves ===
plt.figure(figsize=(10, 6))
for i in range(min(5, NUM_FUNCTIONS)):
    plt.plot(convergence_data[i], label=f"F{i+1}")
plt.xlabel("Evaluations")
plt.ylabel("Best Fitness")
plt.title("Convergence Curves (Sample Functions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fsro_convergence_sample.png")
plt.show()

# === Wilcoxon example (against a dummy competitor) ===
# Replace with real comparison algorithm results
dummy_results = np.random.rand(NUM_RUNS, NUM_FUNCTIONS)
p_values = []

for i in range(NUM_FUNCTIONS):
    stat, p = wilcoxon(results_matrix[:, i], dummy_results[:, i])
    p_values.append(p)

summary["Wilcoxon_p"] = p_values
summary.to_csv("fsro_with_significance.csv", index=False)
