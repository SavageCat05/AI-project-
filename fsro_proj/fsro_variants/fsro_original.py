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


