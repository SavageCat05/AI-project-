# This work belongs to Kanav. 
# Do not change without his permission.
# Differential Evolution Optimization Algorithm.

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    F = 0.5
    CR = 0.9

    X = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(x) for x in X])
    
    convergence_curve = []
    evaluations = pop_size

    while evaluations < max_evals:
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = X[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lower_bounds, upper_bounds)
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, X[i])
            trial_fitness = fobj(trial)
            evaluations += 1

            if trial_fitness < fitness[i]:
                X[i] = trial
                fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        convergence_curve.append(fitness[best_idx])

    best_idx = np.argmin(fitness)
    best_solution = X[best_idx]
    best_fitness = fitness[best_idx]

    return best_solution, best_fitness, convergence_curve
