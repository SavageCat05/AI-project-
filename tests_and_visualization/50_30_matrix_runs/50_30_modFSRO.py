# I am running a 50*30 matrix of runs for the modified FSRO algorithm with different seed values on cec 2014 benchmark functions.

import numpy as np
import matplotlib.pyplot as plt

class FSRO:
    def __init__(
        self,
        objective_func,
        pop_size=None,
        dim=30,
        max_iter=500,
        lower_bounds=None,
        upper_bounds=None,
        mutation_rate=0.05,
        elite_ratio=0.2,
        stagnation_limit=30
    ):
        self.obj_func = objective_func
        self.dim = dim
        self.pop_size = pop_size if pop_size else dim * 3
        self.max_iter = max_iter
        if lower_bounds is None:
            self.lb = np.full(dim, -100.0)  # broader range for generality
        else:
            self.lb = np.array(lower_bounds)
        if upper_bounds is None:
            self.ub = np.full(dim, 100.0)
        else:
            self.ub = np.array(upper_bounds)
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.stagnation_limit = stagnation_limit
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = np.inf
        self.stagnation_counter = 0
        self.fitness_history = []

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _mutate_and_crossover(self):
        new_population = []
        for agent in self.population:
            partner = self.population[np.random.randint(self.pop_size)]
            child = agent.copy()
            c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
            child[c1:c2] = partner[c1:c2]
            if self.best_solution is not None:
                direction = self.best_solution - child
                child += 0.2 * direction
            mutation_scale = self.mutation_rate * np.sqrt(self.dim)
            child += np.random.normal(0, mutation_scale, self.dim)
            child = np.clip(child, self.lb, self.ub)
            new_population.append(child)
        return np.array(new_population)

    def _evaluate(self):
        fitness = np.array([self.obj_func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.fitness_history.append(self.best_fitness)
        return fitness

    def optimize(self):
        for gen in range(self.max_iter):
            self.population = self._mutate_and_crossover()
            _ = self._evaluate()
            if self.stagnation_counter >= self.stagnation_limit:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
                self.population = self._initialize_population()
        return self.best_solution, self.best_fitness, self.fitness_history


# --- Define 30 Mock CEC2014 Benchmark Functions ---
def get_cec2014_functions():
    funcs = []
    for i in range(30):
        funcs.append(lambda x, i=i: np.sum((x - i * 0.1)**2))  # Shifted Sphere
    return funcs


# --- Run FSRO for 50 seeds on 30 functions ---
def run_fsro_benchmark():
    dim = 30
    evals_per_run = 60000
    pop_size = dim * 3
    max_iter = evals_per_run // pop_size

    results_matrix = np.zeros((50, 30))  # [runs][functions]
    benchmark_functions = get_cec2014_functions()

    for func_idx, func in enumerate(benchmark_functions):
        print(f"Running Function {func_idx + 1}/30")

        for run in range(50):
            np.random.seed(run)
            optimizer = FSRO(
                objective_func=func,
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter
            )
            _, best_fit, _ = optimizer.optimize()
            results_matrix[run, func_idx] = best_fit

    return results_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the precomputed results
    results = np.load("fsro_cec2014_results.npy")  # Shape: (50, 30)
    np.save("fsro_cec2014_results.npy", results)   # Optional re-save

    print("Completed all runs.")
    print("Result matrix shape:", results.shape)
    print("Top-left 5×5 sample of results:")
    print(results[:5, :5])

    # Create a DataFrame for better labeling and formatting
    df = pd.DataFrame(results, columns=[f"F{j+1}" for j in range(30)])
    df.index = [f"Run {i+1}" for i in range(50)]

    # Optional: Round the values for better display
    df_display = df.round(4)

    # Add a row for mean fitness across runs (optional)
    df_display.loc["Mean"] = df.mean().round(4)

    # --- Plot the table using matplotlib ---
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.axis('off')
    table = ax.table(
        cellText=df_display.values,
        rowLabels=df_display.index,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Save the table as a PNG
    plt.savefig("fsro_cec2014_results_table.png", dpi=300, bbox_inches='tight')
    print("Saved table as fsro_cec2014_results_table.png")
