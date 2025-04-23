import numpy as np
import sys
import os
from statistics import mean, stdev
import pandas as pd  # Import pandas for easy CSV handling

### warning : the file path is hardcoded, change it to your local path or use relative path, please update the path ## 


# ---------------- Import FSRO & Benchmark Functions ------------------
sys.path.append(os.path.abspath("c:/Users/Anshuman Raj/OneDrive/Desktop/AI project/AI-project-/fsro_proj/fsro_variants"))
from fsro_original import FSRO
sys.path.append(os.path.abspath("c:/Users/Anshuman Raj/OneDrive/Desktop/AI project/AI-project-/tests_and_visualization"))
from test_questions import benchmark_functions

# ---------------- Parameters ------------------
DIM = 30
POP_SIZE = 50
TOTAL_EVALS = 60000
RUNS = 50
MAX_GEN = TOTAL_EVALS // POP_SIZE
BOUNDS = (-100, 100)

# ---------------- Evaluation Matrix Initialization ------------------
num_functions = len(benchmark_functions)
results_matrix = np.zeros((RUNS, num_functions))

# ---------------- Run Benchmark Tests ------------------
for func_idx, func in enumerate(benchmark_functions):
    print(f"\nRunning Benchmark Function {func_idx+1}/{num_functions}: {func.__name__}")
    for run in range(RUNS):
        # Use `run + 1` to create sequential seeds starting from 1
        seed = run + 1  # Sequential seed starting from 1
        np.random.seed(seed)  # Set the seed for reproducibility
        fsro = FSRO(pop_size=POP_SIZE, dim=DIM, max_iter=MAX_GEN, bounds=BOUNDS)  # Removed objective_func argument
        _, best_fit, _ = fsro.optimize()
        results_matrix[run, func_idx] = best_fit
        print(f"  Run {run+1:02d} (Seed {seed:02d}) - Best Fitness: {best_fit:.6f}")

# ---------------- Save Results to CSV ------------------
# Convert the results matrix to a pandas DataFrame for better readability
df_results = pd.DataFrame(results_matrix, columns=[func.__name__ for func in benchmark_functions])

# Save to CSV
output_file = "benchmark_results.csv"
df_results.to_csv(output_file, index=False)
3
# ---------------- Summary Statistics ------------------
print("\nSummary Statistics (Mean ± Std Dev) for Each Function:")
for i, func in enumerate(benchmark_functions):
    col = results_matrix[:, i]
    print(f"{func.__name__:<25}: {mean(col):.6f} ± {stdev(col):.6f}")

# ---------------- Display Final Result Matrix ------------------
print(f"\nFinal 50×{num_functions} Result Matrix (Rows = Runs, Columns = Functions):")
np.set_printoptions(precision=6, suppress=True)
print(results_matrix)
