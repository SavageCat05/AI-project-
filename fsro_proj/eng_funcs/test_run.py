import numpy as np
from fsro_proj.fsro_variants.fsro_modified import FSRO
from eg_funcs import pressure_vessel, spring_design, welded_beam, speed_reducer, gear_train

problems = [
    ("Pressure Vessel", pressure_vessel, 4, (0.0625, 99)),
    ("Spring Design", spring_design, 3, (0.001, 100)),
    ("Welded Beam", welded_beam, 4, (0.1, 10)),
    ("Speed Reducer", speed_reducer, 7, (1, 100)),
    ("Gear Train", gear_train, 4, (12, 60))
]

def run_engineering_with_fsro():
    for name, func, dim, bounds in problems:
        fsro = FSRO(
            objective_func=func,
            dim=dim,
            max_gen=500,
            x_bound=bounds
        )
        best_sol, best_fit, history = fsro.optimize()
        print(f"{name}: Best Fitness = {best_fit:.6f}, Best Solution = {best_sol}")

if __name__ == "__main__":
    run_engineering_with_fsro()
