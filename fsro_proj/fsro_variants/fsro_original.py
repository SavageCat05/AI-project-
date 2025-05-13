#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.
import numpy as np

class FSRO:
    def __init__(self, pop_size=6, dim=2, max_iter=500, bounds=(-5, 5)):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.bounds = bounds
        self.snakes = self._init_agents()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    # Initialize population with random values
    def _init_agents(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))

    # Sphere objective functiong
    def _objective(self, x):
        return np.sum(np.square(x))

    # One iteration of crossover and mutation for exploration
    def _update_agents(self):
        new_snakes = []
        for s in self.snakes:
            partner = self.snakes[np.random.randint(self.pop_size)]
            c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
            child = s.copy()
            child[c1:c2] = partner[c1:c2]
            child += np.random.normal(0, 0.3, self.dim)  # Constant mutation
            child = np.clip(child, self.bounds[0], self.bounds[1])
            new_snakes.append(child)
        self.snakes = np.array(new_snakes)

    # Evaluate all agents and update the global best if needed
    def _evaluate(self):
        fitness = np.array([self._objective(ind) for ind in self.snakes])
        current_best_idx = np.argmin(fitness)
        current_best = self.snakes[current_best_idx]
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < self.best_fitness:
            self.best_solution = current_best
            self.best_fitness = current_best_fitness
        self.history.append(self.best_fitness)

    # Main optimization process
    def optimize(self):
        for _ in range(self.max_iter):
            self._update_agents()
            self._evaluate()
        return self.best_solution, self.best_fitness, self.history

# Example usage:
if __name__ == '__main__':
    fsro = FSRO()
    best_solution, best_fitness, history = fsro.optimize()
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
