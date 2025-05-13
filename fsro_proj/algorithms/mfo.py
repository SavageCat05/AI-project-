# This work belongs to Kanav.  
# Do not change this without his permission.
# Moth Flame Optimization (MFO) Algorithm.

import numpy as np

def optimize(fobj, dim,  lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    moths = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(m) for m in moths])
    
    convergence_curve = []
    evaluations = pop_size

    best_solution = moths[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    while evaluations < max_evals:
        flames = moths[np.argsort(fitness)]
        flame_no = round(pop_size - evaluations * ((pop_size - 1) / max_evals))

        for i in range(pop_size):
            distance = np.abs(flames[i % flame_no] - moths[i])
            b = 1
            t = (np.random.rand(dim) - 0.5) * 2
            moths[i] = distance * np.exp(b * t) * np.cos(2 * np.pi * t) + flames[i % flame_no]
            moths[i] = np.clip(moths[i], lower_bounds, upper_bounds)

        fitness = np.array([fobj(m) for m in moths])
        evaluations += pop_size

        current_best = moths[np.argmin(fitness)]
        current_fitness = np.min(fitness)

        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness

        convergence_curve.append(best_fitness)

    return best_solution, best_fitness, convergence_curve
