#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Problem: Sphere function
def objective(x):
    return np.sum(np.square(x))

# Parameters
POP_SIZE = 10
DIM = 2
MAX_GEN = 500 # Increased iteration count
X_BOUND = (-5, 5)

# Enhancement params
MUTATION_RATE = 0.05
STAGNATION_LIMIT = 20
ELITE_RATIO = 0.2

# Initialize population
def init_agents():
    frogs = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))
    snakes = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))
    return frogs, snakes

frogs, snakes = init_agents()
history = []
iteration = 0
stagnation_counter = 0

# Initialize best solution
best_solution = None
best_fitness = float('inf')

# Setup plot
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("FSRO Exploration vs Exploitation")
ax.set_xlim(X_BOUND)
ax.set_ylim(X_BOUND)
sc_frogs = ax.scatter([], [], c='green', label='Frogs (Exploitation)', marker='o')
sc_snakes = ax.scatter([], [], c='red', label='Snakes (Exploration)', marker='x')
sc_best = ax.scatter([], [], c='blue', label='Best Solution', marker='*', s=100)
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='upper right')

def update(frame):
    global frogs, snakes, iteration, best_solution, best_fitness, MUTATION_RATE, stagnation_counter
    iteration += 1

    # Time-dependent decay factor (with floor to avoid negative/zero mutation)
    decay = max(0.01, 1 - (iteration / MAX_GEN))
    frog_mutation = MUTATION_RATE * decay
    snake_mutation = MUTATION_RATE * 6 * decay

    all_agents = np.vstack((frogs, snakes))
    prev_fitness = np.array([objective(ind) for ind in all_agents])
    prev_best = np.min(prev_fitness)

    # Crossover and mutation for frogs (exploitation)
    new_frogs = []
    for f in frogs:
        partner = frogs[np.random.randint(len(frogs))]
        mask = np.random.rand(DIM) > 0.5
        child = np.where(mask, f, partner)
        child += np.random.normal(0, frog_mutation, DIM)
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_frogs.append(child)
    frogs = np.array(new_frogs)

    # Crossover and mutation for snakes (exploration)
    new_snakes = []
    for s in snakes:
        partner = snakes[np.random.randint(len(snakes))]
        c1, c2 = sorted(np.random.choice(DIM, 2, replace=False))
        child = s.copy()
        child[c1:c2] = partner[c1:c2]
        child += np.random.normal(0, snake_mutation, DIM)
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_snakes.append(child)
    snakes = np.array(new_snakes)

    # Combine all and apply elite selection
    all_agents = np.vstack((frogs, snakes))
    fitness = np.array([objective(ind) for ind in all_agents])
    elites_idx = np.argsort(fitness)[:int(ELITE_RATIO * POP_SIZE)]
    elites = all_agents[elites_idx]

    # Replicator dynamics: adjust frog/snake ratio based on improvement
    imp_frog = np.mean(fitness[:len(frogs)]) - np.mean(prev_fitness[:len(frogs)])
    imp_snake = np.mean(fitness[len(frogs):]) - np.mean(prev_fitness[len(frogs):])
    total_imp = abs(imp_frog) + abs(imp_snake)
    if total_imp == 0:
        r_frog = r_snake = 0.5
    else:
        r_frog = abs(imp_frog) / total_imp

        # Encourage more exploitation over time
        exploration_bias = 1 - (iteration / MAX_GEN)
        r_frog = min(0.9, r_frog + (1 - exploration_bias))  # Increase frog weight
        r_snake = 1 - r_frog

    # Create new population based on dynamic ratios + elites
    n_frogs = max(2, int((POP_SIZE - len(elites)) * r_frog))
    n_snakes = POP_SIZE - len(elites) - n_frogs
    frogs = all_agents[np.random.choice(len(all_agents), n_frogs)]
    snakes = all_agents[np.random.choice(len(all_agents), n_snakes)]
    all_agents = np.vstack((frogs, snakes, elites))

    # Evaluate best
    fitness = np.array([objective(ind) for ind in all_agents])
    current_best_idx = np.argmin(fitness)
    current_best = all_agents[current_best_idx]
    current_best_fitness = fitness[current_best_idx]

    # Update best solution only if a better one is found
    if current_best_fitness < best_fitness:
        best_solution = current_best
        best_fitness = current_best_fitness
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    # Adaptive mutation: increase if stagnating
    if stagnation_counter >= STAGNATION_LIMIT:
        MUTATION_RATE = min(1.0, MUTATION_RATE * 1.5)
        stagnation_counter = 0
        # Reinitialize part of the population to escape local optima
        frogs = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))
        snakes = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))

    # Update plot
    sc_frogs.set_offsets(frogs)
    sc_snakes.set_offsets(snakes)
    sc_best.set_offsets([best_solution])
    iteration_text.set_text(f"Iteration: {iteration} | Best: {best_fitness:.4f}")
    return sc_frogs, sc_snakes, sc_best, iteration_text

ani = FuncAnimation(fig, update, frames=MAX_GEN, interval=200, repeat=False)
plt.show()
