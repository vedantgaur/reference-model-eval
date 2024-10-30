import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Any
import random

# Set random seed for reproducibility
random.seed(42)

# Constants from the provided code
test_models_arena = {
    'zephyr-7b-beta': 1054,
    'qwen1.5-7b-chat': 1070,
    'qwen-14b-chat': 1035,
    'mistral-7b-instruct-v0.3': 1072,
    'zephyr-7b-alpha': 1041,
    'guanaco-33b': 1033,
    'llama-2-13b-chat-hf': 1063,
    'vicuna-13b': 1042,
    'vicuna-7b': 1005,
    'chatglm2-6b': 924
}

def create_random_reference_files(models: List[str]):
    """Create randomized annotation files by sampling from different tiers."""
    for model in models:
        os.makedirs(f'results/final-run/randomized/{model}', exist_ok=True)
        randomized_annotations = []
        for i in range(805):
            rand_tier = random.randint(1, 4)
            tier_file = f'results/final-run/tiered/{model}/tier_{rand_tier}/alpaca_eval_gpt4_turbo_fn/annotations.json'
            try:
                with open(tier_file, 'r') as f:
                    tier_annotations = json.load(f)
                    if i < len(tier_annotations):
                        randomized_annotations.append(tier_annotations[i])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing {tier_file}: {e}")
                continue
        output_file = f'results/final-run/randomized/{model}/randomized_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(randomized_annotations, f, indent=2)
        print(f"Created randomized file for {model} with {len(randomized_annotations)} annotations")

def load_annotations(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return []

def load_leaderboard(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, index_col=0)
        return df
    except FileNotFoundError:
        print(f"Leaderboard file not found: {file_path}")
        return pd.DataFrame()

def bradley_terry(win_matrix: np.ndarray) -> np.ndarray:
    n = win_matrix.shape[0]
    def negloglik(p):
        p = np.exp(p)
        p = p / sum(p)
        nll = 0
        for i in range(n):
            for j in range(i+1, n):
                if win_matrix[i,j] + win_matrix[j,i] > 0:
                    pij = p[i] / (p[i] + p[j])
                    nll -= win_matrix[i,j] * np.log(pij) + win_matrix[j,i] * np.log(1 - pij)
        return nll
    res = minimize(negloglik, np.zeros(n), method='BFGS')
    p = np.exp(res.x)
    return p / sum(p)

def create_win_matrix(annotations: List[Dict[str, Any]], models: List[str]) -> np.ndarray:
    n = len(models)
    win_matrix = np.zeros((n, n))
    for ann in annotations:
        if ann['generator_1'] in models and ann['generator_2'] in models:
            i = models.index(ann['generator_2'])
            j = models.index(ann['generator_1'])
            if ann['preference'] == 2.0:
                win_matrix[i,j] += 1
            elif ann['preference'] == 1.0:
                win_matrix[j,i] += 1
    return win_matrix

def process_tiered_evaluation(model: str, leaderboard: pd.DataFrame, annotations: List[Dict[str, Any]], threshold: float = 0.2) -> Tuple[List[float], int]:
    all_winrates = []
    current_tier = 1
    while current_tier <= 4:
        tier_winrate = leaderboard.loc[f"{model}_tier_{current_tier}", 'win_rate']
        if tier_winrate < threshold:
            current_tier += 1
        else:
            all_winrates.append(tier_winrate)
        if current_tier > 4:
            break
    
    # Create win matrix from annotations for the current model
    win_matrix = create_win_matrix(annotations, [model])  # Pass only the current model
    rankings = bradley_terry(win_matrix)  # Get rankings using Bradley-Terry model
    return all_winrates, current_tier, rankings  # Return rankings as well

def calculate_correlations(scores1: List[float], scores2: List[float]) -> Dict[str, Tuple[float, float]]:
    pearson_corr, pearson_p = pearsonr(scores1, scores2)
    kendall_corr, kendall_p = kendalltau(scores1, scores2)
    spearman_corr, spearman_p = spearmanr(scores1, scores2)
    return {
        'pearson': (pearson_corr, pearson_p),
        'kendall': (kendall_corr, kendall_p),
        'spearman': (spearman_corr, spearman_p)
    }

def extract_win_rates(model_name: str, tier: int) -> Tuple[float, float]:
    file_path = f'results/new-run/tiered/{model_name}/tier_{tier}/alpaca_eval_gpt4_turbo_fn/leaderboard.csv'
    df = pd.read_csv(file_path)

    # Extract the regular win rate
    regular_win_rate = df['win_rate'].values[0]
    
    # Extract the length-controlled win rate
    length_controlled_win_rate = df['length_controlled_winrate'].values[0]
    
    return regular_win_rate, length_controlled_win_rate

def main():
    models = list(test_models_arena.keys())
    arena_scores = [test_models_arena[model] for model in models]
    
    # Load leaderboard data
    leaderboard = load_leaderboard('results/final-run/leaderboard.csv')
    
    # Calculate baseline tier correlations using leaderboard data
    print("\nCalculating baseline tier correlations...")
    for tier in range(1, 5):
        tier_winrates = []
        tier_lc_winrates = []
        for model in models:
            win_rate, lc_win_rate = extract_win_rates(model, tier)
            tier_winrates.append(win_rate)
            tier_lc_winrates.append(lc_win_rate)
        
        correlations = calculate_correlations(arena_scores, tier_winrates)
        print(f"\nTier {tier} correlations (standard winrate):")
        for metric, (corr, p_val) in correlations.items():
            print(f"{metric}: correlation = {corr:.4f}, p-value = {p_val:.4f}")
        
        correlations_lc = calculate_correlations(arena_scores, tier_lc_winrates)
        print(f"\nTier {tier} correlations (length-controlled winrate):")
        for metric, (corr, p_val) in correlations_lc.items():
            print(f"{metric}: correlation = {corr:.4f}, p-value = {p_val:.4f}")
    
    # Process tiered evaluations
    print("\nProcessing tiered evaluations...")
    tiered_winrates = {}
    for model in models:
        winrates, final_tier, rankings = process_tiered_evaluation(model, leaderboard, randomized_annotations)  # Pass annotations
        tiered_winrates[model] = winrates
        print(f"Rankings for {model}: {rankings}")  # Print rankings for each model

    # Create combined win matrix for tiered evaluations
    all_tiered_winrates = []
    for winrates in tiered_winrates.values():
        all_tiered_winrates.extend(winrates)

    # Calculate correlations for tiered winrates
    print("\nTiered Winrate Correlations:")
    correlations = calculate_correlations(arena_scores, all_tiered_winrates)
    for metric, (corr, p_val) in correlations.items():
        print(f"{metric}: correlation = {corr:.4f}, p-value = {p_val:.4f}")
    
    # Process randomized evaluations
    print("\nProcessing randomized evaluations...")
    randomized_annotations = []
    for model in models:
        file_path = f'results/final-run/randomized/{model}/randomized_annotations.json'
        randomized_annotations.extend(load_annotations(file_path))
    
    win_matrix_random = create_win_matrix(randomized_annotations, models)
    random_bt_scores = bradley_terry(win_matrix_random)
    
    # Calculate correlations for randomized Bradley-Terry scores
    print("\nRandomized Bradley-Terry Correlations:")
    correlations = calculate_correlations(arena_scores, random_bt_scores)
    for metric, (corr, p_val) in correlations.items():
        print(f"{metric}: correlation = {corr:.4f}, p-value = {p_val:.4f}")

if __name__ == "__main__":
    main()