import numpy as np

def pressure_vessel(x):
    x1, x2, x3, x4 = x
    cost = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4.0 / 3.0) * np.pi * x3**3 + 1296000
    g4 = x4 - 240
    penalty = sum([max(0, g)**2 for g in [g1, g2, g3, g4]]) * 10_000
    return cost + penalty


def spring_design(x):
    x1, x2, x3 = x
    eps = 1e-8  # Small value to avoid divide-by-zero

    cost = (x3 + 2) * x1 * x2**2

    # Constraint 1: avoid zero in denominator
    denom1 = 71785 * x1**4 if abs(x1) > eps else eps
    g1 = 1 - (x2**3 * x3) / denom1

    # Constraint 2: multiple denominators with potential zero
    denom2 = 12566 * (x2 * x1**3 - x1**4)
    denom2 = denom2 if abs(denom2) > eps else eps
    denom3 = 5108 * x1**2 if abs(x1) > eps else eps
    g2 = (4 * x2**2 - x1 * x2) / denom2 + 1 / denom3 - 1

    # Constraint 3: avoid zero in denominator
    denom4 = x2**2 * x3 if abs(x2) > eps and abs(x3) > eps else eps
    g3 = 1 - (140.45 * x1) / denom4

    g4 = (x1 + x2) - 1.5

    penalty = sum([max(0, g)**2 for g in [g1, g2, g3, g4]]) * 1_000
    return cost + penalty


def welded_beam(x):
    h, l, t, b = x
    P = 6000
    eps = 1e-8

    cost = 1.10471 * h**2 * l + 0.04811 * b * t * (14 + l)

    g1 = -h + 0.125

    denom_tau = np.sqrt(2) * h * l if abs(h) > eps and abs(l) > eps else eps
    tau = P / denom_tau

    denom_sigma = b * t**2 if abs(b) > eps and abs(t) > eps else eps
    sigma = (6 * P * 14) / denom_sigma

    tau_max = 13600
    sigma_max = 30000

    g2 = tau - tau_max
    g3 = sigma - sigma_max

    penalty = sum([max(0, g)**2 for g in [g1, g2, g3]]) * 10_000
    return cost + penalty


def speed_reducer(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    eps = 1e-8

    cost = (
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )

    safe_x2 = x2 if abs(x2) > eps else eps
    safe_x3 = x3 if abs(x3) > eps else eps

    constraints = [
        27 - x1 * x2**2 * x3,
        397.5 - x1 * x2**2 * x3,
        1.93 * x4**3 / (safe_x2 * safe_x3**2) - 1,
        1.93 * x5**3 / (safe_x2 * safe_x3**2) - 1,
        (745 * x4 / (safe_x2 * safe_x3))**2 - 16.9e6,
        (745 * x5 / (safe_x2 * safe_x3))**2 - 16.9e6,
        x2 - x1,
        1.5 - x6,
        1.5 - x7
    ]
    penalty = sum([max(0, g)**2 for g in constraints]) * 10_000
    return cost + penalty


def gear_train(x):
    x1, x2, x3, x4 = np.round(x).astype(int)
    eps = 1e-8
    denom = x3 * x4 if x3 * x4 != 0 else eps
    cost = ((1 / 6.931) - (x1 * x2) / denom)**2
    return cost
