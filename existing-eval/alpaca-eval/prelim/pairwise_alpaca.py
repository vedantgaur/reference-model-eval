import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import time

models = [
    "gpt-4o-2024-05-13", "gpt-3.5-turbo-1106", "llama-2-7b-chat-hf", "llama-2-13b-chat-hf",
    "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", 
    "Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x22B-Instruct-v0.1"
]

runs_dir = "~/projects/llm-eval/existing-eval/alpaca-eval/git-results"

def run_alpaca_eval(reference_model, model):
    command = f"alpaca_eval --reference_outputs '{runs_dir}/{reference_model}/model_outputs.json' --model_outputs '{runs_dir}/{model}/model_outputs.json' --annotators_config 'alpaca_eval_gpt4_turbo_fn'"
    subprocess.run(command, shell=True, check=True)
    time.sleep(2)

    results_file = f"results/{model}/alpaca_eval_gpt4_turbo_fn/leaderboard.csv"

    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            length_controlled_winrate = float(row['length_controlled_winrate']) / 100
            break

    print(f"Length-controlled win rate for {model} vs {reference_model}: {length_controlled_winrate}%")

    return length_controlled_winrate

    
    # full_output = result.stdout + result.stderr
    # print(full_output)
    
    # output_lines = [line.strip() for line in full_output.split('\n') if line.strip()]
    
    # # if not output_lines:
    # #     print(f"No output for {model} vs {reference_model}")
    # #     return 0.0
    
    # last_line = output_lines[-1]
    
    # print(f"Final output for {model} vs {reference_model}:")
    # print(last_line)
    
    # parts = last_line.split(',')
    # win_rate = float(parts[-1]) / 100 
    
    # return win_rate / 100



results = pd.DataFrame(index=models, columns=models)

for reference_model in models:
    for model in models:
        if reference_model != model:
            print(f"Starting {reference_model} vs {model}")
            win_rate = run_alpaca_eval(reference_model, model)
            print(f"Done with {reference_model} vs {model}")
            print("=====================================")
            results.loc[reference_model, model] = win_rate
        else:
            results.loc[reference_model, model] = 0.5 

results.to_csv("~/projects/llm-eval/existing-eval/alpaca-eval/data/pairwise_results.csv")

plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt='.2f')
plt.title("Alpaca Eval Pairwise Model Comparison (Win Rates)")
plt.tight_layout()
plt.savefig("~/projects/llm-eval/existing-eval/alpaca-eval/model_comparison_heatmap.png")
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
plt.savefig("~/projects/llm-eval/existing-eval/alpaca-eval/average_win_rates_barplot.png")
plt.close()

