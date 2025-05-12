
import numpy as np
from fsro_proj.fsro_variants.fsro_modified import FSRO
from opfunu.cec_based.cec2020 import (
    F12020, F22020, F32020, F42020, F52020, F62020, F72020, F82020, F92020, F102020,
    F112020, F122020, F132020, F142020, F152020, F162020, F172020, F182020, F192020, F202020
)

cec2020_funcs = [
    F12020, F22020, F32020, F42020, F52020, F62020, F72020, F82020, F92020, F102020,
    F112020, F122020, F132020, F142020, F152020, F162020, F172020, F182020, F192020, F202020
]

def run_cec2020_with_fsro():
    results = []
    dim = 10
    max_gen = 500
    x_bound = (-100, 100)

    for i, func_class in enumerate(cec2020_funcs, start=1):
        func_instance = func_class(dim=dim)
        fsro = FSRO(
            objective_func=func_instance.evaluate,
            dim=dim,
            max_gen=max_gen,
            x_bound=x_bound
        )
        best_sol, best_fit, history = fsro.optimize()
        print(f"CEC2020 Function F{i:02d}: Best Fitness = {best_fit:.6f}")
        results.append((f"F{i:02d}", best_fit))
    return results

if __name__ == "__main__":
    run_cec2020_with_fsro()
