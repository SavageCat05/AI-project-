import numpy as np

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
        """
        FSRO optimizer scaled for high-dimensional problems.
        """
        self.obj_func = objective_func
        self.dim = dim
        self.pop_size = pop_size if pop_size else dim * 3  # Scale population with dimension
        self.max_iter = max_iter
        # Default bounds if not provided
        if lower_bounds is None:
            self.lb = np.full(dim, -5.0)
        else:
            self.lb = np.array(lower_bounds)

        if upper_bounds is None:
            self.ub = np.full(dim, 5.0)
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
        """
        Randomly initialize the population within bounds.
        """
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _mutate_and_crossover(self):
        """
        Generate new population using crossover, directional movement, and mutation.
        """
        new_population = []
        for agent in self.population:
            partner = self.population[np.random.randint(self.pop_size)]
            child = agent.copy()

            # Crossover: randomly select a segment to copy from partner
            c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
            child[c1:c2] = partner[c1:c2]

            # Directional movement towards best solution (with scale)
            if self.best_solution is not None:
                direction = self.best_solution - child
                child += 0.2 * direction  # movement step

            # Scaled mutation based on dimension
            mutation_scale = self.mutation_rate * np.sqrt(self.dim)
            child += np.random.normal(0, mutation_scale, self.dim)

            # Bound the solution
            child = np.clip(child, self.x_bound[0], self.x_bound[1])
            new_population.append(child)

        return np.array(new_population)

    def _evaluate(self):
        """
        Evaluate current population and update best solution.
        """
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
        """
        Run the FSRO optimization loop.
        """
        for gen in range(self.max_iter):
            self.population = self._mutate_and_crossover()
            _ = self._evaluate()

            # Adapt on stagnation
            if self.stagnation_counter >= self.stagnation_limit:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
                self.population = self._initialize_population()

        return self.best_solution, self.best_fitness, self.fitness_history

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Test on a 30D sphere function (like CEC functions)
    def sphere(x):
        return np.sum(x**2)

    optimizer = FSRO(objective_func=sphere, dim=4)
    best_sol, best_fit, history = optimizer.optimize()

    print("Best Solution:", best_sol)
    print("Best Fitness:", best_fit)

    # Human-understandable interpretation
    print("\n--- Analysis ---")
    print(f"Number of Dimensions: {optimizer.dim}")
    print(f"Target Optimum (for Sphere function): 0.0")
    print(f"Best Fitness Achieved: {best_fit:.6f}")

    # Qualitative assessment
    if best_fit < 1e-5:
        quality = "Excellent (near global optimum)"
    elif best_fit < 1e-2:
        quality = "Very Good"
    elif best_fit < 1.0:
        quality = "Good"
    elif best_fit < 10:
        quality = "Fair"
    else:
        quality = "Needs Improvement"

    print(f"Performance Quality: {quality}")
    print(f"Convergence Speed: {len(history)} generations")

    # Plot convergence curve
    plt.figure(figsize=(12, 8))
    plt.plot(history, label='Best Fitness So Far', color='royalblue', linewidth=2, marker='o', markersize=5, markerfacecolor='red')
    plt.axhline(y=0, color='darkred', linestyle='--', linewidth=2, label="Target Optimum (0.0)")

    # Add text annotations on the graph
    plt.text(0.5, best_fit, f'Best Fitness: {best_fit:.6f}\n{quality}', color='black', fontsize=14,
             ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Labels and Title
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    plt.title("FSRO Convergence Curve on Sphere Function (30D)", fontsize=16)
    plt.yscale("log")  # Log scale for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()