# FSRO Variant 5: Opposition-Based FSRO (O-FSRO)
import numpy as np

class OFSRO:
    def __init__(self, objective_func, pop_size=10, dim=10, max_gen=100, x_bound=(-5, 5)):
        self.obj_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_gen = max_gen
        self.x_bound = x_bound
        self.population = self._init_pop()
        self.best_solution = None
        self.best_fitness = float('inf')

    def _init_pop(self):
        return np.random.uniform(self.x_bound[0], self.x_bound[1], (self.pop_size, self.dim))

    def optimize(self):
        for _ in range(self.max_gen):
            opposite = self.x_bound[0] + self.x_bound[1] - self.population
            combined = np.vstack((self.population, opposite))
            fitness = np.array([self.obj_func(ind) for ind in combined])
            best_idx = np.argmin(fitness)
            self.best_solution = combined[best_idx].copy()
            self.best_fitness = fitness[best_idx]
            top_indices = np.argsort(fitness)[:self.pop_size]
            self.population = combined[top_indices]

        return self.best_solution, self.best_fitness
