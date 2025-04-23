import numpy as np

class FSRO:
    def __init__(self, objective_func, pop_size=6, dim=2, max_gen=100, x_bound=(-5, 5), mutation_rate=0.05, elite_ratio=0.2, stagnation_limit=20):
        # Core FSRO parameters
        self.obj_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_gen = max_gen
        self.x_bound = x_bound
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.stagnation_limit = stagnation_limit

        # Initialize population (snakes only as per new logic)
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.stagnation_counter = 0
        self.fitness_history = []

    def _initialize_population(self):
        return np.random.uniform(self.x_bound[0], self.x_bound[1], (self.pop_size, self.dim))

    def _mutate_and_crossover(self):
        new_population = []
        for agent in self.population:
            partner = self.population[np.random.randint(self.pop_size)]
            c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
            child = agent.copy()
            child[c1:c2] = partner[c1:c2]

            # Move towards the best solution known so far
            if self.best_solution is not None:
                direction = self.best_solution - child
                child += 0.2 * direction

            child += np.random.normal(0, self.mutation_rate * 6, self.dim)
            child = np.clip(child, self.x_bound[0], self.x_bound[1])
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
        for generation in range(self.max_gen):
            self.population = self._mutate_and_crossover()
            fitness = self._evaluate()

            # Adaptive mutation in case of stagnation
            if self.stagnation_counter >= self.stagnation_limit:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
                self.population = self._initialize_population()

        return self.best_solution, self.best_fitness, self.fitness_history

if __name__ == "__main__":
    def objective(x):
        return np.sum(np.square(x))

    fsro = FSRO(objective_func=objective)
    best_sol, best_fit, history = fsro.optimize()
    print("Best Solution:", best_sol)
    print("Best Fitness:", best_fit)