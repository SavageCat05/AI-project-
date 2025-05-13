import numpy as np

D = 10  # All problems now have 10 dimensions

def pressure_vessel(x):
    x = np.asarray(x)
    x1, x2, x3, x4 = x[:4]
    cost = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * x3**2 * x4 - (4.0 / 3.0) * np.pi * x3**3 + 1296000
    g4 = x4 - 240
    penalty = sum([max(0, g)**2 for g in [g1, g2, g3, g4]]) * 10_000
    return cost + penalty

pressure_vessel.lb = np.array([0.01, 0.01, 10.0, 10.0] + [0.0]*(D - 4))
pressure_vessel.ub = np.array([99.0, 99.0, 200.0, 240.0] + [1.0]*(D - 4))

def spring_design(x):
    x = np.asarray(x)
    x1, x2, x3 = x[:3]
    eps = 1e-8
    cost = (x3 + 2) * x1 * x2**2
    denom1 = 71785 * x1**4 if abs(x1) > eps else eps
    g1 = 1 - (x2**3 * x3) / denom1
    denom2 = 12566 * (x2 * x1**3 - x1**4)
    denom2 = denom2 if abs(denom2) > eps else eps
    denom3 = 5108 * x1**2 if abs(x1) > eps else eps
    g2 = (4 * x2**2 - x1 * x2) / denom2 + 1 / denom3 - 1
    denom4 = x2**2 * x3 if abs(x2) > eps and abs(x3) > eps else eps
    g3 = 1 - (140.45 * x1) / denom4
    g4 = (x1 + x2) - 1.5
    penalty = sum([max(0, g)**2 for g in [g1, g2, g3, g4]]) * 1_000
    return cost + penalty

spring_design.lb = np.array([0.05, 0.25, 2.0] + [0.0]*(D - 3))
spring_design.ub = np.array([2.0, 1.3, 15.0] + [1.0]*(D - 3))

def welded_beam(x):
    x = np.asarray(x)
    h, l, t, b = x[:4]
    P = 6000
    eps = 1e-8
    cost = 1.10471 * h**2 * l + 0.04811 * b * t * (14 + l)
    g1 = -h + 0.125
    denom_tau = np.sqrt(2) * h * l if abs(h) > eps and abs(l) > eps else eps
    tau = P / denom_tau
    denom_sigma = b * t**2 if abs(b) > eps and abs(t) > eps else eps
    sigma = (6 * P * 14) / denom_sigma
    g2 = tau - 13600
    g3 = sigma - 30000
    penalty = sum([max(0, g)**2 for g in [g1, g2, g3]]) * 10_000
    return cost + penalty

welded_beam.lb = np.array([0.125, 0.1, 0.1, 0.1] + [0.0]*(D - 4))
welded_beam.ub = np.array([5.0, 10.0, 10.0, 5.0] + [1.0]*(D - 4))

def speed_reducer(x):
    x = np.asarray(x)
    x1, x2, x3, x4, x5, x6, x7 = x[:7]
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

speed_reducer.lb = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0] + [0.0]*(D - 7))
speed_reducer.ub = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5] + [1.0]*(D - 7))

def gear_train(x):
    x = np.round(x).astype(int)
    x1, x2, x3, x4 = x[:4]
    eps = 1e-8
    denom = x3 * x4 if x3 * x4 != 0 else eps
    cost = ((1 / 6.931) - (x1 * x2) / denom)**2
    return cost

gear_train.lb = np.array([12, 12, 12, 12] + [0]*(D - 4))
gear_train.ub = np.array([60, 60, 60, 60] + [1]*(D - 4))
