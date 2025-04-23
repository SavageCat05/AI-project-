import numpy as np
import sys
import os

# ---------------- Path Setup ------------------
# Add the fsro_original module to the system path for import
sys.path.append(os.path.abspath("c:/Users/Anshuman Raj/OneDrive/Desktop/AI project/AI-project-/fsro_proj/fsro_variants"))
from fsro_original import FSRO  # type: ignore # Import the original FSRO class

# ---------------- Speed Reducer Objective Function ------------------
# This defines the objective function for the speed reducer mechanical design problem.
def speed_reducer_objective(x):
    # Unpack variables for readability
    x1, x2, x3, x4, x5, x6, x7 = x

    # Objective function to minimize (cost function)
    return (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.477 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2))

# ---------------- Variable Bounds ------------------
# Each design variable has a specific lower and upper bound.
bounds = [
    (2.6, 3.6),   # x1: face width
    (0.7, 0.8),   # x2: module of teeth
    (17, 28),     # x3: number of teeth
    (7.3, 8.3),   # x4: length of first shaft
    (7.8, 8.3),   # x5: length of second shaft
    (2.9, 3.9),   # x6: thickness of first shaft
    (5.0, 5.5)    # x7: thickness of second shaft
]

# Convert bounds to numpy arrays for use in FSRO
low, high = np.array(bounds).T

# ---------------- FSRO Setup ------------------
# Initialize FSRO with appropriate dimensions and bounds
fsro = FSRO(pop_size=10, dim=7, max_iter=300, bounds=(low, high))

# Override the objective function with the speed reducer problem
fsro._objective = speed_reducer_objective

# ---------------- Run Optimization ------------------
# Start the FSRO optimization
best_solution, best_fitness, history = fsro.optimize()

# ---------------- Output Results ------------------
# Print best solution found and its cost (fitness)
print("Best Solution Found:", best_solution)
print("Best Fitness (Cost):", best_fitness)

# ---------------- Grey Wolf Optimizer Function ------------------
def optimize(fobj, bounds, max_evals):
    dim = len(bounds)
    pop_size = 30
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

# ---------------- Run Optimization ------------------
max_evaluations = 300  # Set the maximum number of evaluations
best_solution, convergence_curve = optimize(speed_reducer_objective, bounds, max_evaluations)

# ---------------- Output Results ------------------
print("Best Solution Found:", best_solution)
print("Best Fitness (Cost):", speed_reducer_objective(best_solution))