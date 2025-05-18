import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import de, gwo, hho, mfo, pso, sma, woa
from fsro_variants import fsro_modified, fsro_original
from opfunu.cec_based import cec2014


def run_experiments(dimension=10, max_evals=60000):
    benchmark_functions, benchmark_labels = [], []

    for i in range(1, 31):
        func_name = f"F{i}2014"
        if hasattr(cec2014, func_name):
            benchmark_functions.append(getattr(cec2014, func_name)(ndim=dimension))
            benchmark_labels.append(f"F{i}_CEC2014")
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

    fitness_scores_data = []  # Only one score per function per algorithm

    for func_idx, func in enumerate(benchmark_functions):
        func_name = benchmark_labels[func_idx]
        print(f"\nRunning on function {func_idx + 1}/{len(benchmark_functions)}: {func_name}")

        for algo_name, algo in algorithms.items():
            print(f"  Running {algo_name}...")

            np.random.seed(1000 + func_idx)  # deterministic seed per function

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
                "Fitness": best_fitness
            })

    # Save single-run fitness scores
    os.makedirs("CEC2014_benchmarks_results", exist_ok=True)
    pd.DataFrame(fitness_scores_data).to_csv('CEC2014_benchmarks_results/one_run_fitness_scores_CEC2014.csv', index=False)

    return fitness_scores_data


def plot_fitness_scores(fitness_data):
    df = pd.DataFrame(fitness_data)
    pivot_df = df.pivot(index='Function', columns='Algorithm', values='Fitness')

    plt.figure(figsize=(14, 6))
    for algo_name in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[algo_name], label=algo_name, marker='o')

    plt.xticks(rotation=90)
    plt.xlabel("Benchmark Functions (CEC2014)")
    plt.ylabel("Fitness Value")
    plt.title("Single-Run Fitness Comparison on 30 CEC2014 Functions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs('CEC2014_benchmarks_results', exist_ok=True)
    plt.savefig('CEC2014_benchmarks_results/one_run_fitness_scores_curve_CEC2014.png')
    # plt.show()


if __name__ == "__main__":
    fitness_data = run_experiments()
    plot_fitness_scores(fitness_data)
