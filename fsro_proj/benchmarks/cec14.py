import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from fsro_variants.fsro_modified import FSRO

from opfunu.cec_based.cec2014 import (
    F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014,
    F112014, F122014, F132014, F142014, F152014, F162014, F172014, F182014, F192014, F202014,
    F212014, F222014, F232014, F242014, F252014, F262014, F272014, F282014, F292014, F302014
)

cec2014_funcs = [
    F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014,
    F112014, F122014, F132014, F142014, F152014, F162014, F172014, F182014, F192014, F202014,
    F212014, F222014, F232014, F242014, F252014, F262014, F272014, F282014, F292014, F302014
]

def run_cec2014_with_fsro():
    results = []
    dim = 10
    max_gen = 500
    x_bound = (-100, 100)

    for i, func_class in enumerate(cec2014_funcs, start=1):
        func_instance = func_class(ndim=dim)
        fsro = FSRO(
            objective_func=func_instance.evaluate,
            dim=dim,
            max_gen=max_gen,
            x_bound=x_bound
        )
        best_sol, best_fit, history = fsro.optimize()
        print(f"CEC2014 Function F{i:02d}: Best Fitness = {best_fit:.6f}")
        results.append((f"F{i:02d}", best_fit))
    return results

if __name__ == "__main__":
    run_cec2014_with_fsro()
