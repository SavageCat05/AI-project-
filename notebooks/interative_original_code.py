#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Problem: Sphere function
def objective(x):
    return np.sum(np.square(x))

# Parameters
POP_SIZE = 6  # Reduced population for clearer visualization
DIM = 2
MAX_GEN = 500 # Increased iteration count
X_BOUND = (-5, 5)

# Initialize population
def init_agents():
    snakes = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE, DIM))
    return snakes

snakes = init_agents()
history = []
iteration = 0

# Initialize best solution
best_solution = None
best_fitness = float('inf')

# Setup plot
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("FSRO Exploration vs Exploitation")
ax.set_xlim(X_BOUND)
ax.set_ylim(X_BOUND)

# Show square at origin (target)
ax.scatter(0, 0, s=200, facecolors='none', edgecolors='black', linewidths=2, label='Target (0,0)')

sc_snakes = ax.scatter([], [], c='red', label='Snakes (Search Agents)', marker='x')
sc_best = ax.scatter([], [], c='blue', label='Best Solution', marker='*', s=100)
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
best_value_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)  # Display best fitness
ax.legend(loc='upper right')

# Convergence data for plotting at the end
convergence_data = []

# Update function for animation
def update(frame):
    global snakes, iteration, best_solution, best_fitness
    iteration += 1

    # Crossover and mutation for snakes (exploration)
    new_snakes = []
    for s in snakes:
        partner = snakes[np.random.randint(len(snakes))]
        c1, c2 = sorted(np.random.choice(DIM, 2, replace=False))
        child = s.copy()
        child[c1:c2] = partner[c1:c2]
        child += np.random.normal(0, 0.3, DIM)  # Constant exploration mutation
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_snakes.append(child)
    snakes = np.array(new_snakes)

    # Evaluate best
    fitness = np.array([objective(ind) for ind in snakes])
    current_best_idx = np.argmin(fitness)
    current_best = snakes[current_best_idx]
    current_best_fitness = fitness[current_best_idx]

    # Update best solution only if a better one is found
    if current_best_fitness < best_fitness:
        best_solution = current_best
        best_fitness = current_best_fitness

    # Save history
    convergence_data.append(best_fitness)

    # Update plot
    sc_snakes.set_offsets(snakes)
    sc_best.set_offsets([best_solution])  # Use the best solution
    iteration_text.set_text(f"Iteration: {iteration}")
    best_value_text.set_text(f"Best Fitness: {best_fitness:.4f}")
    return sc_snakes, sc_best, iteration_text, best_value_text

ani = FuncAnimation(fig, update, frames=MAX_GEN, interval=200, repeat=False)
plt.show()

# Plot convergence graph at the end
plt.figure()
plt.plot(convergence_data, label='Best Fitness')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Convergence Curve')
plt.legend()
plt.grid(True)
plt.show()
