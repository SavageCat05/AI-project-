import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Problem: Sphere function
def objective(x):
    return np.sum(np.square(x))

# Parameters
POP_SIZE = 6  # Reduced number of search agents
DIM = 2
MAX_GEN = 500
X_BOUND = (-5, 5)

# Enhancement params
MUTATION_RATE = 0.05
STAGNATION_LIMIT = 20
ELITE_RATIO = 0.2

# Initialize population
def init_agents():
    return np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE, DIM))

snakes = init_agents()
history = []
iteration = 0
stagnation_counter = 0

# Initialize best solution
best_solution = None
best_fitness = float('inf')

# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("FSRO Exploration vs Exploitation")

# Plot search agents
ax1.set_xlim(X_BOUND)
ax1.set_ylim(X_BOUND)
ax1.scatter(0, 0, s=200, facecolors='none', edgecolors='black', linewidths=2, label='Target (0,0)')
sc_snakes = ax1.scatter([], [], c='red', label='Snakes (Search Agents)', marker='x')
sc_best = ax1.scatter([], [], c='blue', label='Best Solution', marker='*', s=100)
iteration_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
ax1.legend(loc='upper right')

# Plot convergence graph
ax2.set_title('Convergence Curve')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Best Fitness')
convergence_line, = ax2.plot([], [], lw=2, label='Best Fitness')
ax2.legend()
best_fitnesses = []

# Animation update function
def update(frame):
    global snakes, iteration, best_solution, best_fitness, MUTATION_RATE, stagnation_counter
    iteration += 1

    # Time-based and score-based decay: focus more around best after midpoint
    decay = max(0.01, 1 - (iteration / MAX_GEN))
    proximity_weight = min(1.0, iteration / (MAX_GEN * 0.6))
    mutation_strength = MUTATION_RATE * 6 * decay * (1 - 0.7 * proximity_weight)

    new_snakes = []
    for s in snakes:
        partner = snakes[np.random.randint(len(snakes))]
        c1, c2 = sorted(np.random.choice(DIM, 2, replace=False))
        child = s.copy()
        child[c1:c2] = partner[c1:c2]

        # Move towards best solution with some randomness
        if best_solution is not None:
            direction = best_solution - child
            child += 0.2 * proximity_weight * direction

        child += np.random.normal(0, mutation_strength, DIM)
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_snakes.append(child)

    snakes = np.array(new_snakes)

    # Evaluate and select elites
    fitness = np.array([objective(ind) for ind in snakes])
    elites_idx = np.argsort(fitness)[:max(1, int(ELITE_RATIO * POP_SIZE))]
    elites = snakes[elites_idx]

    # Update best solution
    current_best_idx = np.argmin(fitness)
    current_best = snakes[current_best_idx]
    current_best_fitness = fitness[current_best_idx]

    if current_best_fitness < best_fitness:
        best_solution = current_best
        best_fitness = current_best_fitness
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    best_fitnesses.append(best_fitness)

    # Adaptive mutation
    if stagnation_counter >= STAGNATION_LIMIT:
        MUTATION_RATE = min(1.0, MUTATION_RATE * 1.5)
        stagnation_counter = 0
        snakes = init_agents()

    # Update plot data
    sc_snakes.set_offsets(snakes)
    sc_best.set_offsets([best_solution])
    iteration_text.set_text(f"Iteration: {iteration} | Best: {best_fitness:.4f}")

    convergence_line.set_data(range(len(best_fitnesses)), best_fitnesses)
    ax2.relim()
    ax2.autoscale_view()
    return sc_snakes, sc_best, iteration_text, convergence_line

ani = FuncAnimation(fig, update, frames=MAX_GEN, interval=200, repeat=False)
plt.tight_layout()
plt.show()