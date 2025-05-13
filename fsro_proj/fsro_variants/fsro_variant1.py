# FSRO Variant 1: Adaptive Mutation FSRO (AM-FSRO)
import numpy as np

class AM_FSRO:
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

        # Initialize best solution after generating the population
        fitness = np.array([self.obj_func(ind) for ind in self.population])
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()
        
        self.x_bound = (self.lb, self.ub)

    def _init_pop(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
    
    def _adaptive_mutation_rate(self):
        diversity = np.mean([np.linalg.norm(ind - self.best_solution) for ind in self.population])
        return min(1.0, 0.1 + 0.5 * diversity / np.sqrt(self.dim))

    def optimize(self):
        convergence_curve = []

        for _ in range(self.max_iter):
            mutation_rate = self._adaptive_mutation_rate()
            new_pop = []
            for i in range(self.pop_size):
                partner = self.population[np.random.randint(self.pop_size)]
                c1, c2 = sorted(np.random.choice(self.dim, 2, replace=False))
                child = self.population[i].copy()
                child[c1:c2] = partner[c1:c2]
                child += np.random.normal(0, mutation_rate, self.dim)
                child = np.clip(child, self.x_bound[0], self.x_bound[1])
                new_pop.append(child)

            self.population = np.array(new_pop)
            fitness = np.array([self.obj_func(ind) for ind in self.population])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            convergence_curve.append(self.best_fitness)

        return self.best_solution, self.best_fitness, convergence_curve
