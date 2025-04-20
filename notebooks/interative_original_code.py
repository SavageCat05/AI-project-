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

# Initialize population
def init_agents():
    frogs = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))
    snakes = np.random.uniform(X_BOUND[0], X_BOUND[1], (POP_SIZE // 2, DIM))
    return frogs, snakes

frogs, snakes = init_agents()
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
sc_frogs = ax.scatter([], [], c='green', label='Frogs (Exploitation)', marker='o')
sc_snakes = ax.scatter([], [], c='red', label='Snakes (Exploration)', marker='x')
sc_best = ax.scatter([], [], c='blue', label='Best Solution', marker='*', s=100)
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='upper right')

def update(frame):
    global frogs, snakes, iteration, best_solution, best_fitness
    iteration += 1

    # Crossover and mutation for frogs (exploitation)
    new_frogs = []
    for f in frogs:
        partner = frogs[np.random.randint(len(frogs))]
        mask = np.random.rand(DIM) > 0.5
        child = np.where(mask, f, partner)
        child += np.random.normal(0, 0.05, DIM)  # small local mutation
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
        child += np.random.normal(0, 0.3, DIM)  # broader mutation
        child = np.clip(child, X_BOUND[0], X_BOUND[1])
        new_snakes.append(child)
    snakes = np.array(new_snakes)

    # Evaluate best
    all_agents = np.vstack((frogs, snakes))
    fitness = np.array([objective(ind) for ind in all_agents])
    current_best_idx = np.argmin(fitness)
    current_best = all_agents[current_best_idx]
    current_best_fitness = fitness[current_best_idx]

    # Update best solution only if a better one is found
    if current_best_fitness < best_fitness:
        best_solution = current_best
        best_fitness = current_best_fitness

    # Update plot
    sc_frogs.set_offsets(frogs)
    sc_snakes.set_offsets(snakes)
    sc_best.set_offsets([best_solution])  # Use the best solution
    iteration_text.set_text(f"Iteration: {iteration}")
    return sc_frogs, sc_snakes, sc_best, iteration_text

ani = FuncAnimation(fig, update, frames=MAX_GEN, interval=200, repeat=False)
plt.show()
