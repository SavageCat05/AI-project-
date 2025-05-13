# This work belongs to Kanav. 
# dont change this without his permission
# this is code for whale optimization algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 10
    whales = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([fobj(w) for w in whales])
    
    convergence_curve = []
    evaluations = pop_size

    best_whale = whales[np.argmin(fitness)]

    a = 2

    while evaluations < max_evals:
        for i in range(pop_size):
            r = np.random.rand()
            A = 2 * a * np.random.rand(dim) - a
            C = 2 * np.random.rand(dim)
            p = np.random.rand()
            if p < 0.5:
                if np.linalg.norm(A) >= 1:
                    rand_whale = whales[np.random.randint(pop_size)]
                    D = np.abs(C * rand_whale - whales[i])
                    whales[i] = rand_whale - A * D
                else:
                    D = np.abs(C * best_whale - whales[i])
                    whales[i] = best_whale - A * D
            else:
                D = np.abs(best_whale - whales[i])
                whales[i] = D * np.exp(1) * np.cos(2 * np.pi * np.random.rand(dim)) + best_whale

            whales[i] = np.clip(whales[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = np.array([fobj(w) for w in whales])
        evaluations += pop_size
        best_whale = whales[np.argmin(fitness)]
        convergence_curve.append(np.min(fitness))

        a -= 2 / (max_evals / pop_size)

    return best_whale, convergence_curve
