# This work belongs to Kanav. 
# dont change this without his permission
# this is code for smile mould optimization algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 30
    smas = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([fobj(s) for s in smas])
    
    convergence_curve = []
    evaluations = pop_size

    best_sma = smas[np.argmin(fitness)]

    while evaluations < max_evals:
        sorted_idx = np.argsort(fitness)
        for i in range(pop_size):
            r = np.random.rand()
            if r < 0.5:
                r1, r2 = np.random.randint(0, pop_size, 2)
                smas[i] = smas[sorted_idx[0]] + r * (smas[r1] - smas[r2])
            else:
                r1, r2 = np.random.randint(0, pop_size, 2)
                smas[i] = smas[sorted_idx[i]] + r * (smas[r1] - smas[r2])

            smas[i] = np.clip(smas[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = np.array([fobj(s) for s in smas])
        evaluations += pop_size
        best_sma = smas[np.argmin(fitness)]
        convergence_curve.append(np.min(fitness))

    return best_sma, convergence_curve
