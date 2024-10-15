import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr

# Function to extract win rates from leaderboard.csv
def extract_win_rates(model_name, tier):
    file_path = f'results/new-run/tiered/{model_name}/tier_{tier}/alpaca_eval_gpt4_turbo_fn/leaderboard.csv'
    # print(os.path.abspath(file_path))
    df = pd.read_csv(file_path)

    # Extract the regular win rate
    regular_win_rate = df['win_rate'].values[0]
    
    # Extract the length-controlled win rate
    length_controlled_win_rate = df['length_controlled_winrate'].values[0]
    
    return regular_win_rate, length_controlled_win_rate

# List of models to evaluate
models = [
    'llama-2-13b-chat-hf',
    'zephyr-7b-beta',
    'qwen1.5-7b-chat',
    'guanaco-33b',
    'vicuna-13b',
    'zephyr-7b-alpha',
    'qwen-14b-chat',
    'mistral-7b-instruct-v0.3',
    'vicuna-7b',
    'chatglm2-6b'
]

# Initialize dictionaries for scores
test_models_arena = {
    'llama-2-13b-chat-hf': 1063,
    'zephyr-7b-beta': 1054,
    'qwen1.5-7b-chat': 1070,
    'guanaco-33b': 1033,
    'vicuna-13b': 1042,
    'zephyr-7b-alpha': 1041,
    'qwen-14b-chat': 1035,
    'mistral-7b-instruct-v0.3': 1072,
    'vicuna-7b': 1005,
    'chatglm2-6b': 924
}
test_models_alpaca = {}
length_controlled_scores = {}

for i in range(4, 5):
# Extract scores for each model
    for model in models:
        # print(model)
        # {{ edit_1 }} Replace placeholder with actual arena scores
        arena_score = test_models_arena[model]  # Use actual arena scores
        non_length_controlled, length_controlled = extract_win_rates(model, i)
        # print(non_length_controlled, length_controlled)
        
        test_models_arena[model] = arena_score
        test_models_alpaca[model] = non_length_controlled
        length_controlled_scores[model] = length_controlled



    # Calculate Pearson and Spearman correlations
    pearson_corr, pearson_p = pearsonr(list(test_models_arena.values()), list(test_models_alpaca.values()))
    spearman_corr, spearman_p = spearmanr(list(test_models_arena.values()), list(test_models_alpaca.values()))
    length_controlled_pearson_corr, length_controlled_pearson_p = pearsonr(list(test_models_arena.values()), list(length_controlled_scores.values()))
    length_controlled_spearman_corr, length_controlled_spearman_p = spearmanr(list(test_models_arena.values()), list(length_controlled_scores.values()))

    print(f"TIER {i}")
    # Print results
    print("Correlation between ChatBot Arena and Alpaca scores (non-length controlled):")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    print("\nCorrelation between ChatBot Arena and Alpaca scores (length controlled):")
    print(f"Pearson correlation: {length_controlled_pearson_corr:.4f} (p-value: {length_controlled_pearson_p:.4f})")
    print(f"Spearman correlation: {length_controlled_spearman_corr:.4f} (p-value: {length_controlled_spearman_p:.4f})")
    print("===============")
    # # Print the data for verification
    # print("\nModel Scores:")
    # print("Model                    Arena   Alpaca (Non-length controlled)   Alpaca (Length controlled)")
    # print("---------------------------------------------------------------------------------------------")
    # for model in models:
    #     print(f"{model:<22} {test_models_arena[model]:<7} {test_models_alpaca[model]:<7} {length_controlled_scores[model]:<7}")
