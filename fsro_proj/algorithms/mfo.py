# This work belongs to Kanav. 
# dont change this without his permission
# this is code for moth flame optimization algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 10
    moths = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    flames = moths.copy()
    fitness = np.array([fobj(m) for m in moths])
    
    convergence_curve = []
    evaluations = pop_size

    while evaluations < max_evals:
        flames = moths[np.argsort(fitness)]
        flame_no = round(pop_size - evaluations * ((pop_size - 1) / max_evals))
        for i in range(pop_size):
            distance = np.abs(flames[i % flame_no] - moths[i])
            b = 1
            t = (np.random.rand(dim) - 0.5) * 2
            moths[i] = distance * np.exp(b * t) * np.cos(2 * np.pi * t) + flames[i % flame_no]
            moths[i] = np.clip(moths[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = np.array([fobj(m) for m in moths])
        evaluations += pop_size
        convergence_curve.append(np.min(fitness))

    best_solution = moths[np.argmin(fitness)]
    return best_solution, convergence_curve
