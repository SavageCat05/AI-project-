import pandas as pd
from scipy.stats import rankdata, wilcoxon
import os
import pandas as pd

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "combined_csv_for_wilcoxon_analysis.csv")

df = pd.read_csv(file_path)


# Step 1: Per-function summary with ranks
functions = df["Function"].unique()
algorithms = df["Algorithm"].unique()

summary = []
for func in functions:
    func_data = df[df["Function"] == func]
    means = func_data["Mean"].values
    stds = func_data["StdDev"].values
    algs = func_data["Algorithm"].values
    ranks = rankdata(means, method='min')  # lower mean is better
    for alg, mean, std, rank in zip(algs, means, stds, ranks):
        summary.append({
            "Function": func,
            "Algorithm": alg,
            "Mean": mean,
            "StdDev": std,
            "Rank": rank
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("wilcoxon_results/fsro_functionwise_summary.csv", index=False)

# Step 2: Aggregate rankings
rankings = summary_df.groupby("Algorithm")["Rank"].agg(["mean", "sum"]).reset_index()
rankings.columns = ["Algorithm", "AvgRank", "TotalRank"]
rankings = rankings.sort_values("AvgRank")
rankings.to_csv("wilcoxon_results/fsro_algorithm_rankings.csv", index=False)

# Step 3: Wilcoxon signed-rank test against FSRO_Modified
baseline_algo = "FSRO_Modified"
wilcoxon_results = []

for algo in algorithms:
    if algo == baseline_algo:
        continue
    base_means = []
    comp_means = []
    for func in functions:
        base_val = df[(df["Function"] == func) & (df["Algorithm"] == baseline_algo)]["Mean"].values
        comp_val = df[(df["Function"] == func) & (df["Algorithm"] == algo)]["Mean"].values
        if len(base_val) == 1 and len(comp_val) == 1:
            base_means.append(base_val[0])
            comp_means.append(comp_val[0])
    # Perform Wilcoxon test
    if len(base_means) > 0 and len(comp_means) > 0:
        stat, pval = wilcoxon(base_means, comp_means, alternative='less')
        wilcoxon_results.append({
            "ComparedWith": algo,
            "p-value": pval,
            "Significant (p<0.05)": pval < 0.05
        })

wilcoxon_df = pd.DataFrame(wilcoxon_results).sort_values("p-value")
wilcoxon_df.to_csv("wilcoxon_results/fsro_wilcoxon_results_vs_modified.csv", index=False)
