import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import time

# Expand the ~ to the full path
file_path = os.path.expanduser("~/projects/llm-eval/existing-eval/alpaca-eval/data/pairwise_results.csv")
results = pd.read_csv(file_path, index_col=0)

plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt='.2f')
plt.title("Alpaca Eval Pairwise Model Comparison (Win Rates)")
plt.tight_layout()

# Save the heatmap
heatmap_path = os.path.expanduser("~/projects/llm-eval/existing-eval/alpaca-eval/data/model_comparison_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

print(results)

avg_win_rates = results.mean(axis=1).sort_values(ascending=False)
print("\nAverage Win Rates:")
print(avg_win_rates)

plt.figure(figsize=(12, 6))
avg_win_rates.plot(kind='bar')
plt.title("Average Win Rates Across All Comparisons")
plt.xlabel("Models")
plt.ylabel("Average Win Rate")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the bar plot
barplot_path = os.path.expanduser("~/projects/llm-eval/existing-eval/alpaca-eval/data/average_win_rates_barplot.png")
plt.savefig(barplot_path)
plt.close()
