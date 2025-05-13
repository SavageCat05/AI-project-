#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.

import numpy as np
import matplotlib.pyplot as plt

# FSRO algorithm
class FSRO:
    def __init__(self, fobj, pop_size=6, dim=2, max_iter=500, lower_bounds=None, upper_bounds=None):
        self.fobj = fobj
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter

        self.lb = np.full(dim, -5.0) if lower_bounds is None else np.array(lower_bounds)
        self.ub = np.full(dim, 5.0) if upper_bounds is None else np.array(upper_bounds)

        self.snakes = self._init_agents()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def _init_agents(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _objective(self, x):
        return self.fobj(x)

    def _update_agents(self):
        new_snakes = []
        for s in self.snakes:
            partner = self.snakes[np.random.randint(self.pop_size)]
            c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
            child = s.copy()
            child[c1:c2] = partner[c1:c2]
            child += np.random.normal(0, 0.3, self.dim)
            child = np.clip(child, self.lb, self.ub)
            new_snakes.append(child)
        self.snakes = np.array(new_snakes)

    def _evaluate(self):
        fitness = np.array([self._objective(ind) for ind in self.snakes])
        best_idx = np.argmin(fitness)
        best_candidate = self.snakes[best_idx]
        best_fit = fitness[best_idx]

        if best_fit < self.best_fitness:
            self.best_fitness = best_fit
            self.best_solution = best_candidate.copy()

        self.history.append(self.best_fitness)

    def optimize(self):
        for _ in range(self.max_iter):
            self._update_agents()
            self._evaluate()
        return self.best_solution, self.best_fitness, self.history


# ----------------- Main Execution Block ----------------- #
if __name__ == '__main__':
    # Adjustable number of dimensions
    num_dimensions = 10  # You can change this to any number (e.g., 10, 50, 100)
    population_size = num_dimensions * 2  # Scaling pop_size with dimension (optional)

    # Instantiate and run FSRO
    fsro = FSRO(pop_size=population_size, dim=num_dimensions)
    best_solution, best_fitness, history = fsro.optimize()

    # Print results
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    # Human-understandable interpretation
    print("\n--- Analysis ---")
    print(f"Number of Dimensions: {fsro.dim}")
    print(f"Target Optimum (for Sphere function): 0.0")
    print(f"Best Fitness Achieved: {best_fitness:.6f}")

    # Qualitative assessment
    if best_fitness < 1e-5:
        quality = "Excellent (near global optimum)"
    elif best_fitness < 1e-2:
        quality = "Very Good"
    elif best_fitness < 1.0:
        quality = "Good"
    elif best_fitness < 10:
        quality = "Fair"
    else:
        quality = "Needs Improvement"

    print(f"Performance Quality: {quality}")
    print(f"Convergence Speed: {len(history)} iterations")

    # Plot convergence curve
    plt.figure(figsize=(12, 8))
    plt.plot(history, label='Best Fitness So Far', color='royalblue', linewidth=2, marker='o', markersize=5, markerfacecolor='red')
    plt.axhline(y=0, color='darkred', linestyle='--', linewidth=2, label="Target Optimum (0.0)")

    # Text annotation on plot
    plt.text(len(history)*0.5, best_fitness, f'Best Fitness: {best_fitness:.6f}\n{quality}', color='black', fontsize=14,
             ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Labels and Title
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    plt.title(f"FSRO Convergence Curve on Sphere Function ({num_dimensions}D)", fontsize=16)
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
