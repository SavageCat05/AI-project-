# FSRO Variant 2: Leader-Guided FSRO (LG-FSRO)
import numpy as np


class LG_FSRO:
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
            fitness = np.array([self.obj_func(ind) for ind in self.population])
            best_idx = np.argmin(fitness)
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = fitness[best_idx]
            leaders = self.population[np.argsort(fitness)[:max(1, self.pop_size // 3)]]

            new_pop = []
            for i in range(self.pop_size):
                partner = self.population[np.random.randint(self.pop_size)]
                leader = leaders[np.random.randint(len(leaders))]
                child = self.population[i].copy()
                child += 0.3 * (leader - child)
                c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
                child[c1:c2] = partner[c1:c2]
                child += np.random.normal(0, 0.2, self.dim)
                child = np.clip(child, self.x_bound[0], self.x_bound[1])
                new_pop.append(child)

            self.population = np.array(new_pop)

        return self.best_solution, self.best_fitness
