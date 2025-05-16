import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms import de, gwo, hho, mfo, pso, sma, woa
from fsro_variants import (
    fsro_modified, fsro_original
)
from opfunu.cec_based import cec2022


def run_experiments(dimension=10, max_evals=60000, num_runs=10):

    benchmark_functions, benchmark_labels = [], []

    
    for i in range(1, 11):  
        func_name = f"F{i}2022"
        if hasattr(cec2022, func_name):
            benchmark_functions.append(getattr(cec2022, func_name)(ndim=dimension))
            benchmark_labels.append(f"F{i}_CEC2022")
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

    results, convergence_data = [], {}

    for func_idx, func in enumerate(benchmark_functions):
        func_name = benchmark_labels[func_idx]
        print(f"\nRunning on function {func_idx + 1}/{len(benchmark_functions)}: {func_name}")
        convergence_data[func_name] = {}

        for algo_name, algo in algorithms.items():
            print(f"  Running {algo_name}...")
            best_scores, all_convergence = [], []

            for run in tqdm(range(num_runs), desc=f"{algo_name} runs"):
                seed = run + 1000
                np.random.seed(seed)

                if "FSRO" in algo_name:
                    
                    lb = np.array(func.lb)
                    ub = np.array(func.ub)

                    # Pad to match required dimension
                    if len(lb) < dimension:
                        lb = np.concatenate([lb, np.zeros(dimension - len(lb))])
                    if len(ub) < dimension:
                        ub = np.concatenate([ub, np.ones(dimension - len(ub))])
                                        
                    optimizer = algo(func.evaluate, dim=dimension, max_iter=max_evals // dimension,
                                     lower_bounds= lb, upper_bounds= ub)
                    best_solution, best_fitness, fitness_curve = optimizer.optimize()
                else:
                    
                    lb = np.array(func.lb)
                    ub = np.array(func.ub)

                    # Pad to match required dimension
                    if len(lb) < dimension:
                        lb = np.concatenate([lb, np.zeros(dimension - len(lb))])
                    if len(ub) < dimension:
                        ub = np.concatenate([ub, np.ones(dimension - len(ub))])
                        
                    best_solution, _, fitness_curve = algo(
                        func.evaluate,
                        dim=dimension,
                        lower_bounds=lb,
                        upper_bounds=ub,
                        max_evals=max_evals,
                    )
                    best_fitness = func.evaluate(best_solution)

                best_scores.append(best_fitness)
                all_convergence.append(fitness_curve)

            results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                "Mean": np.mean(best_scores),
                "StdDev": np.std(best_scores),
                "Min": np.min(best_scores),
                "Max": np.max(best_scores),
            })
            convergence_data[func_name][algo_name] = np.mean(all_convergence, axis=0)

    os.makedirs("CEC2022_benchmarks_results", exist_ok=True)
    pd.DataFrame(results).to_csv('CEC2022_benchmarks_results/comparison_result_CEC2022.csv', index=False)
    np.save('CEC2022_benchmarks_results/convergence_data.npy', convergence_data)

    return results, convergence_data


def plot_comparison(convergence_data, functions_to_plot=None):
    if functions_to_plot is None:
        functions_to_plot = list(convergence_data.keys())[:]

    for func_name in functions_to_plot:
        plt.figure(figsize=(10, 6))
        for algo_name, data in convergence_data[func_name].items():
            plt.plot(data, label=algo_name)
        plt.title(f"Convergence Curve - {func_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs('CEC2022_benchmarks_results', exist_ok=True)
        plt.savefig(f"CEC2022_benchmarks_results/{func_name}_convergence.png")
        
        # plt.show()


if __name__ == "__main__":
    results, convergence_data = run_experiments()
    plot_comparison(convergence_data)
