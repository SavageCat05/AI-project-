import numpy as np
from fsro_proj.fsro_variants.fsro_modified import FSRO
from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022,
    F112022, F122022, F132022, F142022, F152022, F162022, F172022, F182022, F192022, F202022
)

cec2022_funcs = [
    F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022,
    F112022, F122022, F132022, F142022, F152022, F162022, F172022, F182022, F192022, F202022
]

def run_cec2022_with_fsro():
    results = []
    dim = 10
    max_gen = 500
    x_bound = (-100, 100)

    for i, func_class in enumerate(cec2022_funcs, start=1):
        func_instance = func_class(dim=dim)
        fsro = FSRO(
            objective_func=func_instance.evaluate,
            dim=dim,
            max_gen=max_gen,
            x_bound=x_bound
        )
        best_sol, best_fit, history = fsro.optimize()
        print(f"CEC2022 Function F{i:02d}: Best Fitness = {best_fit:.6f}")
        results.append((f"F{i:02d}", best_fit))
    return results

if __name__ == "__main__":
    run_cec2022_with_fsro()
