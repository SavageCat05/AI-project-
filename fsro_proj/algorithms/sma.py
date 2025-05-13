# This work belongs to Kanav.  
# Do not change this without his permission.
# Slime Mould Algorithm (SMA)

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    X = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(x) for x in X])
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx]
    best_fitness = fitness[best_idx]

    convergence_curve = []
    evaluations = pop_size

    W = np.ones((pop_size, dim))  # Weight matrix

    while evaluations < max_evals:
        S = np.argsort(fitness)
        X = X[S]
        fitness = fitness[S]

        best_solution = X[0]
        best_fitness = fitness[0]

        a = np.tanh(abs(fitness[0] - fitness[-1]))
        b = 1 - evaluations / max_evals

        for i in range(pop_size):
            for j in range(dim):
                r = np.random.rand()
                vb = np.random.rand()
                vc = np.random.rand()
                A = np.sign(r - 0.5) * a * (1 - b)

                p = np.tanh(abs(fitness[i] - best_fitness))
                W[i, j] = p * ((1 - b) + A)

                r1, r2 = np.random.randint(pop_size), np.random.randint(pop_size)
                X[i, j] = best_solution[j] + W[i, j] * (X[r1, j] - X[r2, j])
        
        X = np.clip(X, lower_bounds, upper_bounds)
        fitness = np.array([fobj(x) for x in X])
        evaluations += pop_size

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = X[best_idx]
            best_fitness = fitness[best_idx]

        convergence_curve.append(best_fitness)

    return best_solution, best_fitness, convergence_curve
