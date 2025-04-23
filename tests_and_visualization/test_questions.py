#### NOTE : I am Anshuman and comments have been specifically added to the code to make it more readable and understandable.
#### PLEASE ~ DO NOT REMOVE THE COMMENTS without my permission.

import numpy as np

# 30 Benchmark Functions for Global Optimization
# Standard test suite covering uni-modal and multi-modal landscapes

def sphere(x):
    return np.sum(x ** 2)

def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_1_2(x):
    return np.sum(np.cumsum(x)**2)

def schwefel_2_21(x):
    return np.max(np.abs(x))

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def step(x):
    return np.sum(np.floor(x + 0.5) ** 2)

def quartic_noise(x):
    return np.sum(np.arange(1, len(x)+1) * (x ** 4)) + np.random.rand()

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def griewank(x):
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

def penalized_1(x):
    a = 10
    return np.pi / len(x) * (10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2 +
        np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2)) +
        ((x[-1] + 1) / 4) ** 2) + np.sum(a * (x > 10) * (x - 10) ** 2 + a * (x < -10) * (-x - 10) ** 2)

def penalized_2(x):
    y = 1 + (x + 1) / 4
    return np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2)) + (y[-1] - 1) ** 2 + np.sum(100 * (x > 10) * (x - 10) ** 4 + 100 * (x < -10) * (-x - 10) ** 4)

def michalewicz(x):
    m = 10
    return -np.sum(np.sin(x) * np.sin(np.arange(1, len(x)+1) * x ** 2 / np.pi) ** (2 * m))

def zakharov(x):
    return np.sum(x ** 2) + np.sum(0.5 * np.arange(1, len(x)+1) * x) ** 2 + np.sum(0.5 * np.arange(1, len(x)+1) * x) ** 4

def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

def bent_cigar(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)

def discus(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)

def different_powers(x):
    return np.sum(np.abs(x) ** (2 + 4 * np.arange(len(x)) / (len(x) - 1)))

def noisy_sphere(x):
    return np.sum(x ** 2) + 1e-6 * np.random.randn()

def ellipsoid(x):
    return np.sum(10 ** (6 * np.arange(len(x)) / (len(x) - 1)) * x ** 2)

def rotated_ellipsoid(x):
    # Identity for simplicity in this example
    return ellipsoid(x)

def rotated_bent_cigar(x):
    return bent_cigar(x)

def rotated_discus(x):
    return discus(x)

def rotated_schwefel(x):
    return schwefel(x)

def rotated_rastrigin(x):
    return rastrigin(x)

def rotated_ackley(x):
    return ackley(x)

def rotated_griewank(x):
    return griewank(x)

# List of all benchmark functions
benchmark_functions = [
    sphere, schwefel_2_22, schwefel_1_2, schwefel_2_21, rosenbrock, step, quartic_noise, schwefel, rastrigin,
    ackley, griewank, penalized_1, penalized_2, michalewicz, zakharov, levy, bent_cigar, discus, different_powers,
    noisy_sphere, ellipsoid, rotated_ellipsoid, rotated_bent_cigar, rotated_discus, rotated_schwefel,
    rotated_rastrigin, rotated_ackley, rotated_griewank
]
