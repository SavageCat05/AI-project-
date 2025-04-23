# This work belongs to Kanav. 
# dont change this without his permission
# this is code for defferential evolution optimization algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 30
    F = 0.5
    CR = 0.9

    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([fobj(x) for x in X])
    
    convergence_curve = []
    evaluations = pop_size

    while evaluations < max_evals:
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = X[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
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

    best_solution = X[np.argmin(fitness)]
    return best_solution, convergence_curve
