# This work belongs to Kanav.  
# Do not change this without his permission.
# Particle Swarm Optimization (PSO) Algorithm.

import numpy as np

def optimize(fobj, dim, lower_bounds, upper_bounds, max_evals):
    pop_size = 10
    w = 0.5
    c1 = 1.5
    c2 = 1.5

    X = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    V = np.zeros_like(X)

    pbest = X.copy()
    pbest_scores = np.array([fobj(x) for x in X])
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)

    convergence_curve = []
    evaluations = pop_size

    while evaluations < max_evals:
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V

        for i in range(pop_size):
            X[i] = np.clip(X[i], lower_bounds, upper_bounds)
            fit = fobj(X[i])
            evaluations += 1

            if fit < pbest_scores[i]:
                pbest[i] = X[i]
                pbest_scores[i] = fit

                if fit < gbest_score:
                    gbest = X[i]
                    gbest_score = fit

        convergence_curve.append(gbest_score)

    return gbest, gbest_score, convergence_curve
