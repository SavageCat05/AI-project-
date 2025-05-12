# FSRO Variant 4: Chaotic FSRO (CFSRO)
import numpy as np

class CFSRO:
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

    def _chaotic_seq(self):
        r = 4
        x = np.random.rand()
        seq = []
        for _ in range(self.dim):
            x = r * x * (1 - x)
            seq.append(x)
        return np.array(seq)

    def optimize(self):
        for _ in range(self.max_gen):
            fitness = np.array([self.obj_func(ind) for ind in self.population])
            best_idx = np.argmin(fitness)
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = fitness[best_idx]

            new_pop = []
            for ind in self.population:
                chaotic = self._chaotic_seq()
                child = ind + chaotic * 0.5
                child = np.clip(child, self.x_bound[0], self.x_bound[1])
                new_pop.append(child)

            self.population = np.array(new_pop)

        return self.best_solution, self.best_fitness