import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import all FSRO variants
from fsro_variants import (
    fsro_modified, fsro_original, fsro_variant1,
    fsro_variant2, fsro_variant3, fsro_variant4, fsro_variant5
)

# Import benchmark algorithms
from algorithms import de, gwo, hho, mfo, pso, sma, woa

# Import benchmark functions from opfunu
from opfunu.cec_based import cec2014, cec2017, cec2020, cec2022

# Import engineering problems
from eng_funcs.eg_func import pressure_vessel, spring_design, welded_beam, speed_reducer, gear_train


# Main experiment running function
def run_experiments(dimension=10, max_evals=60000, num_runs=50):
    
    #Step 1A: Prepare engineering problems
    engineering_problems = {
        "Pressure_Vessel": {
            "func": pressure_vessel,
            "dim": 4,
            "lb": np.array([0.0625, 0.0625, 10, 10]),
            "ub": np.array([10, 10, 200, 240])
        },
        "Spring_Design": {
            "func": spring_design,
            "dim": 3,
            "lb": np.array([0.05, 0.25, 2.0]),
            "ub": np.array([2.0, 1.3, 15.0])
        },
        "Welded_Beam": {
            "func": welded_beam,
            "dim": 4,
            "lb": np.array([0.1, 0.1, 0.1, 0.1]),
            "ub": np.array([2.0, 10.0, 10.0, 2.0])
        },
        "Speed_Reducer": {
            "func": speed_reducer,
            "dim": 7,
            "lb": np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]),
            "ub": np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
        },
        "Gear_Train": {
            "func": gear_train,
            "dim": 4,
            "lb": np.array([12, 12, 12, 12]),
            "ub": np.array([60, 60, 60, 60])
        }
    }
    
    # Step 1B: Prepare benchmark functions
    benchmark_functions = []
    benchmark_labels = []
    
    # CEC 2014
    for i in range(1, 14):
        func_class = getattr(cec2014, f"F{i}2014")
        benchmark_functions.append(func_class(ndim=dimension))
        benchmark_labels.append(f"F{i}_CEC2014")
        
    
    # CEC 2017
    for i in range(1, 10):
        func_class = getattr(cec2017, f"F{i}2017")
        benchmark_functions.append(func_class(ndim=dimension))
        benchmark_labels.append(f"F{i}_CEC2017")
        
    # CEC 2020
    for i in range(1, 5):
        func_class = getattr(cec2020, f"F{i}2020")
        benchmark_functions.append(func_class(ndim=dimension))
        benchmark_labels.append(f"F{i}_CEC2020")
        
    # CEC 2022
    for i in range(1, 5):
        func_class = getattr(cec2022, f"F{i}2022")
        benchmark_functions.append(func_class(ndim=dimension))
        benchmark_labels.append(f"F{i}_CEC2022")
        
    # Integrate engineering problems
    for name, meta in engineering_problems.items():
        # Wrap func to provide same interface as CEC functions
        class Wrapper:
            def __init__(self, evaluate, lb, ub):
                self.evaluate = evaluate
                self.lb = lb
                self.ub = ub
        benchmark_functions.append(Wrapper(meta["func"], meta["lb"], meta["ub"]))
        benchmark_labels.append(name)

    # Step 2: Define all optimization algorithms to test
    algorithms = {
        "FSRO_Original": fsro_original.FSRO,
        "FSRO_Modified": fsro_modified.FSRO,
        "FSRO_Variant1": fsro_variant1.AM_FSRO,
        "FSRO_Variant2": fsro_variant2.LG_FSRO,
        "FSRO_Variant3": fsro_variant3.PSO_FSRO,
        "FSRO_Variant4": fsro_variant4.CFSRO,
        "FSRO_Variant5": fsro_variant5.OFSRO,
        "DE": de.optimize,
        "GWO": gwo.optimize,
        "HHO": hho.optimize,
        "MFO": mfo.optimize,
        "PSO": pso.optimize,
        "SMA": sma.optimize,
        "WOA": woa.optimize,
    }

    # Step 3: Prepare paths and result containers
    results_path = 'comparison_results.csv'
    convergence_path = 'convergence_data.npy'
    results = []
    convergence_data = {}

    # Step 4: Loop through each benchmark function
    for func_idx, func in enumerate(benchmark_functions):
        func_name = benchmark_labels[func_idx]
        print(f"\nRunning on function {func_idx + 1}/{len(benchmark_functions)}: {func_name}")
        convergence_data[func_name] = {}

        # Step 5: Loop through each algorithm
        for algo_name, algo in algorithms.items():
            print(f"  Running {algo_name}...")
            best_scores = []
            all_convergence = []

            # Step 6: Repeat for multiple independent runs
            for run in tqdm(range(num_runs), desc=f"{algo_name} runs"):
                seed = run + 1000
                np.random.seed(seed)

                if "FSRO" in algo_name:
                    optimizer = algo(func.evaluate, dim=dimension, max_iter=max_evals // dimension, 
                                     lower_bounds=func.lb, upper_bounds=func.ub)

                    best_solution, best_fitness, fitness_curve = optimizer.optimize() 
                    best_scores.append(best_fitness)
                    all_convergence.append(fitness_curve)
                else:
                    best_solution, _,  convergence_curve = algo(
                        func.evaluate,
                        dim = dimension,
                        lower_bounds = np.full(dimension, func.lb),
                        upper_bounds = np.full(dimension, func.ub),
                        max_evals = max_evals,
                    )
                    best_scores.append(func.evaluate(best_solution))
                    all_convergence.append(convergence_curve)

            # Step 7: Save statistical performance
            results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                "Mean": np.mean(best_scores),
                "StdDev": np.std(best_scores),
                "Min": np.min(best_scores),
                "Max": np.max(best_scores),
            })

            # Step 8: Save convergence curve data
            convergence_data[func_name][algo_name] = np.mean(all_convergence, axis=0)

    # Step 9: Write results to disk
    pd.DataFrame(results).to_csv(results_path, index=False)
    np.save(convergence_path, convergence_data)

    return results, convergence_data


# Plotting function for visual comparison
def plot_comparison(convergence_data, functions_to_plot=None):
    if functions_to_plot is None:
        functions_to_plot = list(convergence_data.keys())[:5]  # Default: plot first 5 functions

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
        plt.savefig(f"{func_name}_convergence.png")
        plt.show()


# Entry point
if __name__ == "__main__":
    results, convergence_data = run_experiments()
    plot_comparison(convergence_data)
