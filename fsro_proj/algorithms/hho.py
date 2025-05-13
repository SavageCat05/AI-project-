# This work belongs to Kanav. 
# dont change this without his permission
# this is code for harris hawks optimization algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 10
    hawks = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([fobj(h) for h in hawks])
    
    convergence_curve = []
    evaluations = pop_size

    best_hawk = hawks[np.argmin(fitness)]

    while evaluations < max_evals:
        E1 = 2 * (1 - evaluations / max_evals)
        for i in range(pop_size):
            E0 = 2 * np.random.rand() - 1
            E = E1 * E0

            if abs(E) >= 1:
                q = np.random.rand()
                rand_hawk = hawks[np.random.randint(pop_size)]
                hawks[i] = rand_hawk - np.random.rand(dim) * abs(rand_hawk - 2 * np.random.rand(dim) * hawks[i])
            else:
                r = np.random.rand()
                if r >= 0.5 and abs(E) < 0.5:
                    hawks[i] = best_hawk - E * abs(best_hawk - hawks[i])
                else:
                    jump_strength = 2 * (1 - np.random.rand())
                    hawks[i] = best_hawk - E * abs(jump_strength * best_hawk - hawks[i])

            hawks[i] = np.clip(hawks[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = np.array([fobj(h) for h in hawks])
        evaluations += pop_size
        best_hawk = hawks[np.argmin(fitness)]
        convergence_curve.append(np.min(fitness))

    return best_hawk, convergence_curve
