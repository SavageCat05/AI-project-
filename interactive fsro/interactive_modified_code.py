import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective Function
def sphere(x):
    return np.sum(np.square(x))

# Configurations
POP_SIZE = 6
DIM = 2
MAX_GEN = 300
X_BOUND = (-5, 5)

MUTATION_RATE = 0.05
STAGNATION_LIMIT = 20
ELITE_RATIO = 0.2

# Initialize population
def init_population():
    return np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE, DIM))

# Global state
population = init_population()
best_solution = None
best_fitness = float("inf")
stagnation_counter = 0
fitness_history = []
iteration = 0

# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("FSRO Interactive Visualization")

# Left: Search space
ax1.set_xlim(X_BOUND)
ax1.set_ylim(X_BOUND)
ax1.set_title("Agent Movement in Search Space")
ax1.scatter(0, 0, s=200, facecolors='none', edgecolors='black', linewidths=2, label="Global Optimum")
sc_agents = ax1.scatter([], [], c='red', label='Agents', marker='x')
sc_best = ax1.scatter([], [], c='blue', label='Best', marker='*', s=100)
iteration_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
ax1.legend(loc='upper right')

# Right: Convergence
ax2.set_title("Convergence Curve")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness (Log Scale)")
line_fit, = ax2.plot([], [], lw=2, label='Best Fitness')
ax2.set_yscale('log')
ax2.legend()

# Update function
def update(frame):
    global population, best_solution, best_fitness, stagnation_counter, iteration
    iteration += 1
    decay = max(0.01, 1 - (iteration / MAX_GEN))
    proximity_weight = min(1.0, iteration / (MAX_GEN * 0.6))
    mutation_strength = MUTATION_RATE * 6 * decay * (1 - 0.7 * proximity_weight)

    new_population = []
    shuffled = np.random.permutation(population)

    for i, agent in enumerate(population):
        partner = shuffled[i]
        child = agent.copy()

        # Crossover
        c1, c2 = sorted(np.random.choice(DIM, 2, replace=False))
        child[c1:c2] = partner[c1:c2]

        # Directional movement
        if best_solution is not None:
            direction = best_solution - child
            child += 0.2 * proximity_weight * direction

        # Mutation
        child += np.random.normal(0, mutation_strength, DIM)
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_population.append(child)

    population = np.array(new_population)

    # Evaluate
    fitness = np.array([sphere(ind) for ind in population])
    min_idx = np.argmin(fitness)
    current_best_fit = fitness[min_idx]

    if current_best_fit < best_fitness:
        best_fitness = current_best_fit
        best_solution = population[min_idx].copy()  # FIXED: copy the best solution
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    fitness_history.append(best_fitness)

    # Adaptive reinitialization (partial or full)
    if stagnation_counter >= STAGNATION_LIMIT:
        stagnation_counter = 0
        # Reinitialize worst half
        worst_idx = np.argsort(fitness)[POP_SIZE // 2:]
        population[worst_idx] = init_population()[0:len(worst_idx)]

    # Plot update
    sc_agents.set_offsets(population)
    sc_best.set_offsets([best_solution])
    iteration_text.set_text(f"Iteration: {iteration} | Best Fitness: {best_fitness:.4e}")
    line_fit.set_data(range(len(fitness_history)), fitness_history)
    ax2.relim()
    ax2.autoscale_view()

    return sc_agents, sc_best, iteration_text, line_fit

ani = FuncAnimation(fig, update, frames=MAX_GEN, interval=200, repeat=False)
plt.tight_layout()
plt.show()
