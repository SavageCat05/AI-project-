# This work belongs to Kanav.  
# Do not change this without his permission.
# Whale Optimization Algorithm (WOA)

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    whales = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(w) for w in whales])
    best_idx = np.argmin(fitness)
    best_solution = whales[best_idx]
    best_fitness = fitness[best_idx]

    convergence_curve = []
    evaluations = pop_size

    while evaluations < max_evals:
        a = 2 - evaluations * (2 / max_evals)

        for i in range(pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r

            p = np.random.rand()
            l = np.random.uniform(-1, 1)

            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_solution - whales[i])
                    whales[i] = best_solution - A * D
                else:
                    rand_whale = whales[np.random.randint(pop_size)]
                    D = abs(C * rand_whale - whales[i])
                    whales[i] = rand_whale - A * D
            else:
                D = abs(best_solution - whales[i])
                whales[i] = D * np.exp(0.5 * l) * np.cos(2 * np.pi * l) + best_solution

            whales[i] = np.clip(whales[i], lower_bounds, upper_bounds)

        fitness = np.array([fobj(w) for w in whales])
        evaluations += pop_size

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_solution = whales[current_best_idx]
            best_fitness = fitness[current_best_idx]

        convergence_curve.append(best_fitness)

    return best_solution, best_fitness, convergence_curve
