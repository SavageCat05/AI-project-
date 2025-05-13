# This work belongs to Kanav. 
# dont change this without his permission
# this is code for gwo algorithm.

import numpy as np

def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 10
    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([fobj(x) for x in X])
    
    alpha, beta, delta = X[np.argsort(fitness)[:3]]
    convergence_curve = []
    evaluations = pop_size

    a = 2

    while evaluations < max_evals:
        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            X[i] = (X1 + X2 + X3) / 3
            X[i] = np.clip(X[i], [b[0] for b in bounds], [b[1] for b in bounds])

        fitness = np.array([fobj(x) for x in X])
        evaluations += pop_size
        idx = np.argsort(fitness)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        convergence_curve.append(fitness[idx[0]])

        a -= 2 / (max_evals / pop_size)

    return alpha, convergence_curve
