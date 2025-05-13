# FSRO Variant 4: Chaotic FSRO (CFSRO)
import numpy as np

class CFSRO:
    def __init__(self, objective_func, pop_size=10, dim=10, max_iter=100, lower_bounds=None, upper_bounds=None):
        self.obj_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
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
            
        self.population = self._init_pop()
        self.best_solution = None
        self.best_fitness = float('inf')

        fitness = np.array([self.obj_func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()
        
        self.x_bound = (self.lb, self.ub)

    def _init_pop(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _chaotic_seq(self):
        r = 4
        x = np.random.rand()
        seq = []
        for _ in range(self.dim):
            x = r * x * (1 - x)
            seq.append(x)
        return np.array(seq)

    def optimize(self):
        convergence_curve = []

        for _ in range(self.max_iter):
            fitness = np.array([self.obj_func(ind) for ind in self.population])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_solution = self.population[best_idx].copy()
                self.best_fitness = fitness[best_idx]

            new_pop = []
            for ind in self.population:
                chaotic = self._chaotic_seq()
                child = ind + chaotic * 0.5
                child = np.clip(child, self.x_bound[0], self.x_bound[1])
                new_pop.append(child)

            self.population = np.array(new_pop)
            convergence_curve.append(self.best_fitness)

        return self.best_solution, self.best_fitness, convergence_curve