import os
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, kendalltau, spearmanr
from typing import List, Dict, Tuple

# Constants
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/new-run/"
RANDOMIZED_PATH = os.path.join(OUT_PATH, "randomized")
TIERED_PATH = os.path.join(OUT_PATH, "tiered")

# Model lists
all_models = [
    "GPT-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "Yi-34B-Chat",
    "gpt4_1106_preview", "claude-3-opus-20240229", "Llama-3-Instruct-8B-SimPO-ExPO",
    "Llama-3-Instruct-8B-SimPO", "Meta-Llama-3-70B-Instruct", "Meta-Llama-3-8B-Instruct",
    "Qwen1.5-72B-Chat", "gpt4_0314", "claude-3-sonnet-20240229",
    "Mixtral-8x22B-Instruct-v0.1", "gpt4_0613", "Contextual-KTO-Mistral-PairRM",
    "Mistral-7B-Instruct-v0.2", "OpenHermes-2.5-Mistral-7B", "Qwen1.5-7B-Chat",
    "TempNet-LLaMA2-Chat-70B-v0.1", "TempNet-LLaMA2-Chat-13B-v0.1",
    "TempNet-LLaMA2-Chat-7B-v0.1", "alpaca-7b", "alpaca-farm-ppo-human",
    "alpaca-farm-ppo-sim-gpt4-20k", "falcon-40b-instruct", "falcon-7b-instruct"
]

test_models = [
    'llama-2-13b-chat-hf', 'zephyr-7b-beta', 'qwen1.5-7b-chat', 'guanaco-33b',
    'vicuna-13b', 'zephyr-7b-alpha', 'qwen-14b-chat', 'mistral-7b-instruct-v0.3',
    'vicuna-7b', 'chatglm2-6b'
]

tiered_models = [
    "gpt-4-turbo-2024-04-09", "vicuna-33b-v1.3", "llama-2-7b-chat-hf", "oasst-sft-pythia-12b"
]

all_models = [model for model in all_models if model not in test_models and model not in tiered_models]

# Test models and their ChatBot Arena scores
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

def load_annotations(model: str, evaluation_type: str, tier: int = None) -> List[Dict]:
    if evaluation_type == "randomized":
        file_path = os.path.join(RANDOMIZED_PATH, model, "alpaca_eval_gpt4_turbo_fn", "annotations.json")
    elif evaluation_type == "tiered":
        file_path = os.path.join(TIERED_PATH, model, f"tier_{tier}", "alpaca_eval_gpt4_turbo_fn", "annotations.json")
    else:
        raise ValueError("Invalid evaluation type")

    # print(f"Loading annotations from: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        # print(f"Loaded {len(annotations)} annotations for {model}")
        return annotations
    except FileNotFoundError:
        print(f"Annotation file for {model} not found. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return []

def bradley_terry(win_matrix: np.ndarray) -> np.ndarray:
    n = win_matrix.shape[0]
    
    def negloglik(p):
        p = np.exp(p)
        p = p / sum(p)
        nll = 0
        for i in range(n):
            for j in range(i+1, n):
                nll -= win_matrix[i,j] * np.log(p[i] / (p[i] + p[j]))
                nll -= win_matrix[j,i] * np.log(p[j] / (p[i] + p[j]))
        return nll
    
    res = minimize(negloglik, np.zeros(n), method='BFGS')
    p = np.exp(res.x)
    return p / sum(p)

def create_win_matrix(annotations: List[Dict], models: List[str]) -> np.ndarray:
    n = len(models)
    win_matrix = np.zeros((n, n))
    
    for annotation in annotations:
        if annotation['generator_2'] in models and annotation['generator_1'] in models:
            model_index = models.index(annotation['generator_2'])
            reference_index = models.index(annotation['generator_1'])
            if annotation['preference'] == 2.0:
                win_matrix[model_index, reference_index] += 1
            else:
                win_matrix[reference_index, model_index] += 1
    
    return win_matrix

def calculate_rankings(annotations: List[Dict], models: List[str]) -> List[Tuple[str, float]]:
    win_matrix = create_win_matrix(annotations, models)
    bt_rankings = bradley_terry(win_matrix)
    
    return list(zip(models, bt_rankings))

def tiered_ranking(model_results: List[int], tiers: int = 4, threshold: float = 0.2, chunk_size: int = 50) -> Tuple[int, float]:
    current_tier = 1
    total_wins = 0
    total_comparisons = 0
    
    for i in range(0, len(model_results), chunk_size):
        chunk = model_results[i:i+chunk_size]
        chunk_wins = sum(chunk)
        chunk_comparisons = len(chunk)
        
        win_rate = chunk_wins / chunk_comparisons if chunk_comparisons > 0 else 0
        
        if win_rate < threshold and current_tier < tiers:
            current_tier += 1
        elif win_rate >= threshold:
            break
        
        total_wins += chunk_wins
        total_comparisons += chunk_comparisons
    
    final_win_rate = total_wins / total_comparisons if total_comparisons > 0 else 0
    return current_tier, final_win_rate

def process_tiered_results(model: str, threshold: float, chunk_size: int = 50) -> Tuple[int, float]:
    all_results = []
    for tier in range(1, 5):
        annotations = load_annotations(model, "tiered", tier)
        results = [1 if a['preference'] == 2.0 else 0 for a in annotations]
        all_results.extend(results)
    
    return tiered_ranking(all_results, threshold=threshold, chunk_size=50)

def correlation_with_arena(our_rankings: List[float], arena_rankings: List[float]) -> Tuple[float, float, float, float, float, float]:
    pearson_corr, pearson_p = pearsonr(our_rankings, arena_rankings)
    kendall_corr, kendall_p = kendalltau(our_rankings, arena_rankings)
    spearman_corr, spearman_p = spearmanr(our_rankings, arena_rankings)
    return pearson_corr, pearson_p, kendall_corr, kendall_p, spearman_corr, spearman_p

def main():
    # Process randomized rankings
    all_randomized_annotations = []
    for model in test_models:
        annotations = load_annotations(model, "randomized")
        all_randomized_annotations.extend(annotations)

    if not all_randomized_annotations:
        print("Error: No randomized annotations found for any model. Check file paths and content.")
        return

    randomized_rankings = calculate_rankings(all_randomized_annotations, test_models + all_models)
    print("\nRandomized Rankings (Bradley-Terry):")
    for model, score in sorted(randomized_rankings, key=lambda x: x[1], reverse=True):
        if model in test_models:
            print(f"{model}: {score:.4f}")

    # Process tiered rankings for multiple thresholds
    chunk_sizes = [25, 50, 75, 100, 125]
    threshold = 0.2
    for chunk_size in chunk_sizes:
        print(f"\nTiered Rankings (Chunk Size: {chunk_size}):")
        tiered_annotations = []
        tiered_results = []
        for model in test_models:
            tier, win_rate = process_tiered_results(model, threshold, chunk_size)
            tiered_results.append((model, tier, win_rate))
            for t in range(1, tier + 1):
                tiered_annotations.extend(load_annotations(model, "tiered", t))

        tiered_bt_rankings = calculate_rankings(tiered_annotations, test_models + tiered_models)
        
        for model, score in sorted(tiered_bt_rankings, key=lambda x: x[1], reverse=True):
            if model in test_models:
                print(f"{model}: {score:.4f}")

        # Correlation with ChatBot Arena
        arena_rankings = [test_models_arena[model] for model in test_models]
        
        randomized_bt_scores = [score for model, score in randomized_rankings if model in test_models]
        tiered_bt_scores = [score for model, score in tiered_bt_rankings if model in test_models]
        
        print(f"\nCorrelation with ChatBot Arena (Chunk: {chunk_size} // Threshold: {threshold}):")
        
        randomized_pearson, randomized_pearson_p, randomized_kendall, randomized_kendall_p, randomized_spearman, randomized_spearman_p = correlation_with_arena(randomized_bt_scores, arena_rankings)
        print(f"Randomized (Bradley-Terry): Pearson = {randomized_pearson:.4f} (p={randomized_pearson_p:.4f}), Kendall Tau = {randomized_kendall:.4f} (p={randomized_kendall_p:.4f}), Spearman = {randomized_spearman:.4f} (p={randomized_spearman_p:.4f})")
        
        tiered_pearson, tiered_pearson_p, tiered_kendall, tiered_kendall_p, tiered_spearman, tiered_spearman_p = correlation_with_arena(tiered_bt_scores, arena_rankings)
        print(f"Tiered (Bradley-Terry): Pearson = {tiered_pearson:.4f} (p={tiered_pearson_p:.4f}), Kendall Tau = {tiered_kendall:.4f} (p={tiered_kendall_p:.4f}), Spearman = {tiered_spearman:.4f} (p={tiered_spearman_p:.4f})")

if __name__ == "__main__":
    main()
