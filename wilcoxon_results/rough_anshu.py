import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 0: Setup path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "fsro_proj", "wilcoxon_results", "fsro_functionwise_summary.csv")
output_csv = os.path.join(script_dir, "fsro_proj", "wilcoxon_results", "reshaped_functionwise_summary.csv")
plot_dir = os.path.join(script_dir, "function_plots")

# Load the CSV
df = pd.read_csv(input_csv)

# Step 1: Extract 'Function_Name' and 'Year' from 'Function'
df[['Function_Name', 'Year']] = df['Function'].str.extract(r'(F\d+)_(CEC\d{4})')

# Step 2: Pivot the table to get mean, std, and rank side-by-side for each algorithm
pivot_df = df.pivot_table(
    index=['Function_Name', 'Year'],
    columns='Algorithm',
    values=['Mean', 'StdDev', 'Rank']
)

# Step 3: Flatten multi-level columns
pivot_df.columns = [f'{alg}_{metric}' for metric, alg in pivot_df.columns]
pivot_df.reset_index(inplace=True)

# Step 4: Save the reshaped CSV
pivot_df.to_csv(output_csv, index=False)

# Step 5: Generate plots directory
os.makedirs(plot_dir, exist_ok=True)

# Step 6: Generate plots for each Function-Year
algorithms = sorted(set(col.replace('_Mean', '') for col in pivot_df.columns if '_Mean' in col))

for _, row in pivot_df.iterrows():
    func = row['Function_Name']
    year = row['Year']
    means = [row.get(f'{algo}_Mean', None) for algo in algorithms]

    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, means)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean Value")
    plt.title(f"{func}_{year} - Algorithm Mean Comparison")
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"{func}_{year}_mean_comparison.png")
    plt.savefig(plot_path)
    plt.close()
