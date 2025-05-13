# This work belongs to Kanav.  
# Do not change this without his permission.
# Harris Hawks Optimization (HHO) Algorithm.

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    hawks = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    fitness = np.array([fobj(h) for h in hawks])
    
    convergence_curve = []
    evaluations = pop_size

    best_idx = np.argmin(fitness)
    best_hawk = hawks[best_idx]
    best_fitness = fitness[best_idx]

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

            hawks[i] = np.clip(hawks[i], lower_bounds, upper_bounds)

        fitness = np.array([fobj(h) for h in hawks])
        evaluations += pop_size

        current_best_idx = np.argmin(fitness)
        current_best = hawks[current_best_idx]
        current_fitness = fitness[current_best_idx]

        if current_fitness < best_fitness:
            best_hawk = current_best
            best_fitness = current_fitness

        convergence_curve.append(best_fitness)

    return best_hawk, best_fitness, convergence_curve
