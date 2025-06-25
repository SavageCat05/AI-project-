# FSRO Optimization Suite

## Overview

This repository implements and advances the **FSRO (Frog Snake Prey Predation) Optimization Algorithm**, a nature-inspired metaheuristic designed for complex optimization problems. The FSRO algorithm, along with its powerful modifications and variants, is benchmarked against state-of-the-art algorithms on both standard test suites (CEC2014, CEC2017, CEC2020, CEC2022) and real-world engineering design problems.

---

## Features

- **Original and Modified FSRO Algorithms**: Includes the baseline FSRO and a robust, scalable modified version with enhanced convergence and stagnation handling.
- **FSRO Variants**: Five innovative variants (adaptive mutation, leader-guided, hybrid PSO, chaotic, and opposition-based) for diverse optimization landscapes.
- **Comprehensive Benchmarking**: Automated experiments on 30+ CEC benchmark functions and 5 classic engineering design problems.
- **Comparison with Heuristic Algorithms**: Competes with DE, GWO, HHO, MFO, PSO, SMA, and WOA.
- **Statistical Analysis**: Wilcoxon tests and ranking for rigorous performance validation.
- **Rich Visualization**: Convergence curves, result tables, and summary plots for in-depth analysis.
- **Interactive Exploration**: Scripts and notebooks for hands-on experimentation and algorithm understanding.

---

## Project Structure

- `fsro_proj/algorithms/`: Implementations of 7 baseline heuristic algorithms (DE, GWO, HHO, MFO, PSO, SMA, WOA).
- `fsro_proj/fsro_variants/`: Original, modified, and 5 advanced FSRO variants.
- `fsro_proj/eng_funcs/eg_func.py`: Definitions of 5 engineering optimization problems:
  - Pressure Vessel
  - Spring Design
  - Welded Beam
  - Speed Reducer
  - Gear Train
- `fsro_proj/results/`: Benchmark results, convergence data, and plots for all algorithms and functions.
- `eng_results/`: Engineering problem results and convergence plots.
- `wilcoxon_results/`: Statistical test results and algorithm rankings.
- `wilcoxon_test_code/`: Scripts and data for Wilcoxon analysis.
- `tests_and_visualization/`: Test scripts, matrix run utilities, and visualization tools.
- `interactive fsro/`: Interactive scripts for visualizing FSRO convergence.
- `notebooks/`: Summary results and (optionally) Jupyter notebooks for exploration.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/AI-project-.git
   cd AI-project-
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`

---

## Usage & Workflow

### 1. **Running Benchmarks**
- Use `fsro_proj/comparison_curves.py` to run all algorithms on CEC benchmarks.
- Results are saved as CSVs and PNGs in `fsro_proj/results/`.

### 2. **Engineering Problem Optimization**
- Use `fsro_proj/engineering_problem_results.py` to benchmark algorithms on engineering problems.
- Results and convergence plots are saved in `eng_results/`.

### 3. **Statistical Analysis**
- Run Wilcoxon tests using scripts in `wilcoxon_test_code/`.
- Analyze rankings and statistical significance in `wilcoxon_results/`.

### 4. **Matrix Testing**
- `tests_and_visualization/50_30_matrix_runs/50_30_modFSRO.py` runs 50x30 matrix tests for robustness analysis.

### 5. **Interactive Exploration**
- Use scripts in `interactive fsro/` for step-by-step visualization of FSRO convergence.

### 6. **Summary and Visualization**
- Review summary CSVs and plots in `notebooks/` and `fsro_proj/results/`.

---

## FSRO Algorithm & Variants

- **Original FSRO**: Baseline predator-prey inspired optimizer.
- **Modified FSRO**: Scalable, robust, with adaptive mutation and stagnation escape.
- **Variant 1 (Adaptive Mutation)**: Mutation rate adapts to population diversity.
- **Variant 2 (Leader-Guided)**: Exploits best individuals for faster convergence.
- **Variant 3 (Hybrid PSO-FSRO)**: Integrates PSO velocity updates for exploration.
- **Variant 4 (Chaotic FSRO)**: Uses chaotic sequences for enhanced diversity.
- **Variant 5 (Opposition-Based)**: Employs opposition-based learning for global search.

---

## Benchmarking & Testing

- **CEC Benchmarks**: 30+ functions from CEC2014, CEC2017, CEC2020, CEC2022.
- **Engineering Problems**: Five classic design problems with real-world constraints.
- **Matrix Runs**: 50 independent runs per function for statistical robustness.
- **Statistical Tests**: Wilcoxon signed-rank test for pairwise algorithm comparison.
- **Visualization**: Convergence curves, result tables, and summary plots.

---

## Results & Achievements

- **Superior Performance**: The Modified FSRO consistently outperforms both the original and most state-of-the-art algorithms across benchmarks.
- **Statistical Dominance**: Ranked **#1** in Wilcoxon analysis (`wilcoxon_results/fsro_algorithm_rankings.csv`), with the lowest average rank (1.33).
- **Robustness**: Demonstrates low standard deviation and high reliability across 50x30 matrix runs.
- **Engineering Excellence**: Achieves best or near-best solutions on all five engineering design problems.
- **Visualization**: Rich set of convergence plots and result tables for all benchmarks and problems.

---

## Applications

- **Engineering Design**: Pressure vessel, spring, welded beam, speed reducer, and gear train optimization.
- **Global Optimization**: Suitable for high-dimensional, multi-modal, and constrained problems.
- **Algorithm Research**: A testbed for developing and benchmarking new metaheuristics.
- **Education**: Interactive scripts and clear structure for learning optimization techniques.

---

## Testing Criteria

- **Convergence Speed**: Number of iterations to reach near-optimal solutions.
- **Solution Quality**: Mean, min, max, and standard deviation of best fitness values.
- **Statistical Significance**: Wilcoxon test for pairwise comparison.
- **Robustness**: Performance consistency across multiple runs and functions.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**For more details, see the code, results, and documentation in each subfolder.** 