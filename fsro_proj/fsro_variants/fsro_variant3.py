# FSRO Variant 3: Hybrid FSRO-PSO (PSO-FSRO)
import numpy as np


class PSO_FSRO:
    def __init__(self, objective_func, pop_size=10, dim=10, max_gen=100, x_bound=(-5, 5)):
        self.obj_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_gen = max_gen
        self.x_bound = x_bound
        self.population = self._init_pop()
        self.velocity = np.random.uniform(-1, 1, (pop_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def _init_pop(self):
        return np.random.uniform(self.x_bound[0], self.x_bound[1], (self.pop_size, self.dim))

    def optimize(self):
        for _ in range(self.max_gen):
            fitness = np.array([self.obj_func(ind) for ind in self.population])
            best_idx = np.argmin(fitness)
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = fitness[best_idx]

            for i in range(self.pop_size):
                self.velocity[i] = 0.7 * self.velocity[i] + 0.3 * (self.best_solution - self.population[i])
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], self.x_bound[0], self.x_bound[1])

        return self.best_solution, self.best_fitness
