# This work belongs to Kanav. 
# Do not change without his permission.
# Grey Wolf Optimizer (GWO) Algorithm.

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10

    X = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(x) for x in X])
    
    idx = np.argsort(fitness)
    alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
    alpha_fitness = fitness[idx[0]]

    convergence_curve = []
    evaluations = pop_size
    a = 2  # Linearly decreased

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
            X[i] = np.clip(X[i], lower_bounds, upper_bounds)

        fitness = np.array([fobj(x) for x in X])
        evaluations += pop_size

        idx = np.argsort(fitness)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        alpha_fitness = fitness[idx[0]]
        convergence_curve.append(alpha_fitness)

        a -= 2 / (max_evals / pop_size)  # Linearly decreasing 'a'

    best_solution = alpha
    best_fitness = alpha_fitness

    return best_solution, best_fitness, convergence_curve
