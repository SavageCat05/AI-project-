import numpy as np
import sys
import os

# ---------------- Path Setup ------------------
# Add the fsro_original module to the system path for import
sys.path.append(os.path.abspath("c:/Users/Anshuman Raj/OneDrive/Desktop/AI project/AI-project-/fsro_proj/fsro_variants"))
from fsro_modified import FSRO  # type: ignore # Import the original FSRO class

# ---------------- Speed Reducer Objective ------------------
def speed_reducer_objective(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.477 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2))

# ---------------- Bounds Setup ------------------
bounds = [
    (2.6, 3.6),   # x1
    (0.7, 0.8),   # x2
    (17, 28),     # x3
    (7.3, 8.3),   # x4
    (7.8, 8.3),   # x5
    (2.9, 3.9),   # x6
    (5.0, 5.5)    # x7
]
low, high = np.array(bounds).T

# ---------------- Run FSRO on Speed Reducer ------------------
fsro = FSRO(objective_func=speed_reducer_objective, pop_size=10, dim=7, max_gen=300, x_bound=(low, high))
best_solution, best_fitness, history = fsro.optimize()

# ---------------- Results ------------------
print("Best Solution Found:", best_solution)
print("Best Fitness (Cost):", best_fitness)