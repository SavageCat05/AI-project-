import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import de, gwo, hho, mfo, pso, sma, woa
from fsro_variants import fsro_modified, fsro_original
from opfunu.cec_based import cec2017  # Import CEC2017 functions


def run_experiments(dimension=10, max_evals=60000, num_runs=1):
    benchmark_functions, benchmark_labels = [], []

    for i in range(1, 31):
        func_name = f"F{i}2017"
        if hasattr(cec2017, func_name):
            benchmark_functions.append(getattr(cec2017, func_name)(ndim=dimension))
            benchmark_labels.append(f"F{i}_CEC2017")
        else:
            break

    algorithms = {
        "FSRO_Original": fsro_original.FSRO,
        "FSRO_Modified": fsro_modified.FSRO,
        "DE": de.optimize,
        "GWO": gwo.optimize,
        "HHO": hho.optimize,
        "MFO": mfo.optimize,
        "PSO": pso.optimize,
        "SMA": sma.optimize,
        "WOA": woa.optimize,
    }

    fitness_scores_data = []

    for func_idx, func in enumerate(benchmark_functions):
        func_name = benchmark_labels[func_idx]
        print(f"\nRunning on function {func_idx + 1}/{len(benchmark_functions)}: {func_name}")

        for algo_name, algo in algorithms.items():
            print(f"  Running {algo_name}...")

            for run in range(num_runs):  # Still using num_runs, but only one iteration
                np.random.seed(1000 + func_idx)

                lb = np.array(func.lb)
                ub = np.array(func.ub)

                if len(lb) < dimension:
                    lb = np.concatenate([lb, np.zeros(dimension - len(lb))])
                if len(ub) < dimension:
                    ub = np.concatenate([ub, np.ones(dimension - len(ub))])

                if "FSRO" in algo_name:
                    optimizer = algo(
                        func.evaluate,
                        dim=dimension,
                        max_iter=max_evals // dimension,
                        lower_bounds=lb,
                        upper_bounds=ub
                    )
                    best_solution, best_fitness, _ = optimizer.optimize()
                else:
                    best_solution, _, _ = algo(
                        func.evaluate,
                        dim=dimension,
                        lower_bounds=lb,
                        upper_bounds=ub,
                        max_evals=max_evals,
                    )
                    best_fitness = func.evaluate(best_solution)

                fitness_scores_data.append({
                    "Function": func_name,
                    "Algorithm": algo_name,
                    "Run": run + 1,
                    "Fitness": best_fitness
                })

    # Save to CSV
    os.makedirs("CEC2017_benchmarks_results", exist_ok=True)
    pd.DataFrame(fitness_scores_data).to_csv(
        'CEC2017_benchmarks_results/one_run_fitness_scores_CEC2017.csv', index=False)

    return fitness_scores_data


def plot_fitness_scores(fitness_data):
    df = pd.DataFrame(fitness_data)
    pivot_df = df.pivot(index='Function', columns='Algorithm', values='Fitness')

    plt.figure(figsize=(14, 6))
    for algo_name in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[algo_name], label=algo_name, marker='o')

    plt.xticks(rotation=90)
    plt.xlabel("Benchmark Functions (CEC2017)")
    plt.ylabel("Fitness Value")
    plt.title("Single-Run Fitness Comparison on 30 CEC2017 Functions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs('CEC2017_benchmarks_results', exist_ok=True)
    plt.savefig('CEC2017_benchmarks_results/one_run_fitness_scores_curve_CEC2017.png')
    # plt.show()


if __name__ == "__main__":
    fitness_data = run_experiments()
    plot_fitness_scores(fitness_data)
