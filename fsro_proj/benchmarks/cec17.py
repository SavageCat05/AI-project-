import numpy as np
from fsro_proj.fsro_variants.fsro_modified import FSRO
from opfunu.cec_based.cec2017 import (
    F12017, F22017, F32017, F42017, F52017, F62017, F72017, F82017, F92017, F102017,
    F112017, F122017, F132017, F142017, F152017, F162017, F172017, F182017, F192017, F202017,
    F212017, F222017, F232017, F242017, F252017, F262017, F272017, F282017, F292017, F302017
)

cec2017_funcs = [
    F12017, F22017, F32017, F42017, F52017, F62017, F72017, F82017, F92017, F102017,
    F112017, F122017, F132017, F142017, F152017, F162017, F172017, F182017, F192017, F202017,
    F212017, F222017, F232017, F242017, F252017, F262017, F272017, F282017, F292017, F302017
]

def run_cec2017_with_fsro():
    results = []
    dim = 10
    max_gen = 500
    x_bound = (-100, 100)

    for i, func_class in enumerate(cec2017_funcs, start=1):
        func_instance = func_class(dim=dim)
        fsro = FSRO(
            objective_func=func_instance.evaluate,
            dim=dim,
            max_gen=max_gen,
            x_bound=x_bound
        )
        best_sol, best_fit, history = fsro.optimize()
        print(f"CEC2017 Function F{i:02d}: Best Fitness = {best_fit:.6f}")
        results.append((f"F{i:02d}", best_fit))
    return results

if __name__ == "__main__":
    run_cec2017_with_fsro()
