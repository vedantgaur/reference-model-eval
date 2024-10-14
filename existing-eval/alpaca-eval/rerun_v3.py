import os
import json
import random
import subprocess
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_QUESTIONS = 805
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/new-run/"
RANDOMIZED_PATH = os.path.join(OUT_PATH, "randomized")
TIERED_PATH = os.path.join(OUT_PATH, "tiered")

# Available models
all_models = [
    "Conifer-7B-DPO",
    "Contextual-KTO-Mistral-PairRM",
    "Ein-70B-v0.1",
    "FsfairX-Zephyr-Chat-v0.1",
    "LMCocktail-10.7B-v1",
    "Llama-3-Instruct-8B-SimPO-ExPO",
    "Llama-3-Instruct-8B-SimPO",
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B+RAHF-DUAL+LoRA",
    "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-ReMax-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nanbeige-Plus-Chat-v0.1",
    "Nanbeige2-8B-Chat",
    "OpenHermes-2.5-Mistral-7B",
    "Qwen-14B-Chat",
    "Qwen1.5-1.8B-Chat",
    "Qwen1.5-110B-Chat",
    "Qwen1.5-14B-Chat",
    "Qwen1.5-72B-Chat",
    "Qwen1.5-7B-Chat",
    "REBEL-Llama-3-8B-Instruct",
    "SPPO-Mistral7B-PairRM-ExPO",
    "SPPO-Mistral7B-PairRM",
    "Samba-CoE-v0.1",
    "Samba-CoE-v0.2-best-of-16",
    "Samba-CoE-v0.2",
    "Snorkel-Mistral-PairRM-DPO-best-of-16",
    "Snorkel-Mistral-PairRM-DPO",
    "Starling-LM-7B-alpha-ExPO",
    "Starling-LM-7B-alpha",
    "Starling-LM-7B-beta-ExPO",
    "Storm-7B-best-of-64",
    "Storm-7B",
    "TempNet-LLaMA2-Chat-13B-v0.1",
    "TempNet-LLaMA2-Chat-70B-v0.1",
    "TempNet-LLaMA2-Chat-7B-v0.1",
    "Yi-34B-Chat",
    "airoboros-33b",
    "airoboros-65b",
    "aligner-2b_claude-3-opus-20240229",
    "aligner-2b_gpt-4-turbo-2024-04-09",
    "aligner-2b_qwen1.5-72b-chat",
    "alpaca-7b-neft",
    "alpaca-7b",
    "alpaca-7b_concise",
    "alpaca-7b_verbose",
    "alpaca-farm-ppo-human",
    "alpaca-farm-ppo-sim-gpt4-20k",
    "baichuan-13b-chat",
    "baize-v2-13b",
    "baize-v2-7b",
    "bedrock_claude",
    "causallm-14b",
    "chatglm2-6b",
    "claude-2.1",
    "claude-2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-instant-1.2",
    "claude",
    "claude2-alpaca-13b",
    "cohere",
    "cut-13b",
    "dbrx-instruct",
    "deepseek-llm-67b-chat",
    "deita-7b-v1.0",
    "dolphin-2.2.1-mistral-7b",
    "evo-7b",
    "evo-v2-7b",
    "falcon-40b-instruct",
    "falcon-7b-instruct",
    "gemini-pro",
    "gemma-2b-it",
    "gemma-7b-it",
    "ghost-7b-alpha",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt35_turbo_instruct",
    "gpt4",
    "gpt4_0314",
    "gpt4_0613",
    "gpt4_1106_preview",
    "gpt4_gamed"
]

# Test models (subset of all_models, close in performance)
test_models = [
    'vicuna-13b', 
    'wizardlm-13b', 
    'guanaco-33b', 
    'vicuna-7b', 
    'oasst-sft-pythia-12b', 
    'llama-2-13b-chat-hf', 
    'chatglm2-6b'
]

# Tiered models (subset of all_models, not overlapping with test_models)
tiered_models = [
    "gpt-4-0125-preview",
    "claude-3-opus-20240229",
    "claude-2.1",
    "gpt-3.5-turbo-1106",
    "claude-instant-1.2"
]

# Remove test_models and tiered_models from all_models for randomized evaluation
all_models = [model for model in all_models if model not in test_models and model not in tiered_models]

def load_model_outputs(model: str) -> List[Dict]:
    model_output_file = os.path.join(RESULTS_PATH, f"{model}/model_outputs.json")
    try:
        with open(model_output_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Output file for {model} not found. Skipping.")
        return []

def prepare_randomized_reference_outputs(all_models: List[str], num_questions: int, output_file: str) -> None:
    randomized_outputs = []
    for i in range(num_questions):
        random_model = random.choice(all_models)
        model_outputs = load_model_outputs(random_model)
        if model_outputs and i < len(model_outputs):
            output = model_outputs[i]
            output['generator'] = random_model
            randomized_outputs.append(output)
        else:
            logging.warning(f"Skipping question {i} due to missing data for model {random_model}")

    with open(output_file, 'w') as f:
        json.dump(randomized_outputs, f, indent=2)
    logging.info(f"Randomized reference outputs saved to {output_file}")

def prepare_tiered_reference_outputs(tiered_models: List[str], num_questions: int, output_dir: str) -> List[str]:
    tier_files = []
    for i, model in enumerate(tiered_models):
        outputs = []
        model_outputs = load_model_outputs(model)
        for j in range(num_questions):
            if model_outputs and j < len(model_outputs):
                output = model_outputs[j]
                output['generator'] = model
                outputs.append(output)
            else:
                logging.warning(f"Skipping question {j} due to missing data for model {model}")
        
        output_file = os.path.join(output_dir, f"tiered_reference_outputs_{i+1}.json")
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        tier_files.append(output_file)
        logging.info(f"Tiered reference outputs for tier {i+1} saved to {output_file}")
    
    return tier_files

def run_alpaca_eval(model_outputs: str, reference_outputs: str, output_path: str) -> str:
    cmd = [
        "alpaca_eval",
        "--model_outputs", model_outputs,
        "--reference_outputs", reference_outputs,
        "--annotators_config", "alpaca_eval_gpt4_turbo_fn",
        "--output_path", output_path
    ]
    logging.info(f"Running alpaca_eval with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"alpaca_eval completed. Stdout: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running alpaca_eval: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise
    

def run_alpaca_eval_leaderboard(all_model_outputs, reference_outputs, leaderboard_path, annotators_config):
    cmd = [
        "alpaca_eval", "make_leaderboard",
        "--leaderboard_path", leaderboard_path,
        "--all_model_outputs", all_model_outputs,
        "--reference_outputs", reference_outputs,
        "--annotators_config", annotators_config
    ]
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running alpaca_eval make_leaderboard: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise

def get_win_rate(output_path: str, stdout: str) -> float:
    try:
        # First, try to read from the leaderboard.json file
        leaderboard_file = os.path.join(output_path, 'leaderboard.json')
        if os.path.exists(leaderboard_file):
            with open(leaderboard_file, 'r') as f:
                leaderboard = json.load(f)
            return leaderboard[list(leaderboard.keys())[0]]['win_rate']
        else:
            # If the file doesn't exist, parse the stdout
            logging.warning(f"leaderboard.json not found in {output_path}. Parsing stdout.")
            lines = stdout.strip().split('\n')
            if len(lines) > 1:
                last_line = lines[-1].split()
                if len(last_line) >= 3:
                    return float(last_line[2])  # win_rate is the third column
            logging.error(f"Unable to extract win rate from stdout: {stdout}")
            return 0.0
    except Exception as e:
        logging.error(f"Error extracting win rate: {e}")
        return 0.0

def run_tiered_evaluation(test_model: str, tier_files: List[str]) -> Tuple[List[Dict], Dict[str, float]]:
    results = []
    win_rates = defaultdict(list)
    model_outputs = load_model_outputs(test_model)
    model_dir = os.path.join(TIERED_PATH, test_model)
    os.makedirs(model_dir, exist_ok=True)
    model_output_file = os.path.join(model_dir, f"{test_model}_output.json")
    with open(model_output_file, 'w') as f:
        json.dump(model_outputs, f, indent=2)

    for i, reference_outputs in enumerate(tier_files):
        output_path = os.path.join(model_dir, f"tier_{i+1}")
        os.makedirs(output_path, exist_ok=True)
        stdout = run_alpaca_eval(model_output_file, reference_outputs, output_path)

        win_rate = get_win_rate(output_path, stdout)
        results.append({
            'test_model': test_model,
            'reference_model': f'tier_{i + 1}',  # Add this line
            'tier': i + 1,
            'win_rate': win_rate
        })
        win_rates[f"tier_{i + 1}"].append(win_rate)

    avg_win_rates = {tier: sum(rates) / len(rates) for tier, rates in win_rates.items()}
    return results, avg_win_rates

def run_randomized_evaluation(test_model: str, reference_outputs: str) -> Tuple[List[Dict], Dict[str, float]]:
    model_outputs = load_model_outputs(test_model)
    model_dir = os.path.join(RANDOMIZED_PATH, test_model)
    os.makedirs(model_dir, exist_ok=True)
    model_output_file = os.path.join(model_dir, f"{test_model}_output.json")
    with open(model_output_file, 'w') as f:
        json.dump(model_outputs, f, indent=2)

    output_path = os.path.join(model_dir, "alpaca_results")
    os.makedirs(output_path, exist_ok=True)
    stdout = run_alpaca_eval(model_output_file, reference_outputs, output_path)

    win_rate = get_win_rate(output_path, stdout)
    results = [{
        'test_model': test_model,
        'reference_model': 'randomized',  # Add this line
        'win_rate': win_rate
    }]
    avg_win_rates = {test_model: win_rate}
    return results, avg_win_rates

def save_results(results: List[Dict], avg_win_rates: Dict[str, float], test_model: str, eval_type: str) -> None:
    output_dir = os.path.join(OUT_PATH, eval_type, test_model)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'avg_win_rates': avg_win_rates
        }, f, indent=2)
    logging.info(f"Results saved to {output_file}")

def bradley_terry_model(win_matrix):
    n = win_matrix.shape[0]
    
    def neg_log_likelihood(params):
        theta = np.exp(params)
        ll = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    ll += win_matrix[i, j] * np.log(theta[i] / (theta[i] + theta[j]))
        return -ll
    
    def gradient(params):
        theta = np.exp(params)
        grad = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    grad[i] += win_matrix[i, j] - (win_matrix[i, j] + win_matrix[j, i]) * theta[i] / (theta[i] + theta[j])
        return -grad
    
    initial_params = np.zeros(n)
    result = minimize(neg_log_likelihood, initial_params, method='BFGS', jac=gradient)
    return np.exp(result.x)

def compute_elo_rankings(results: List[Dict]) -> pd.DataFrame:
    models = list(set(result['test_model'] for result in results))
    models += list(set(result.get('reference_model') for result in results if 'reference_model' in result))
    models = list(set(models)) 
    n = len(models)
    win_matrix = np.zeros((n, n))

    for result in results:
        i = models.index(result['test_model'])
        if 'reference_model' in result:
            j = models.index(result['reference_model'])
            win_matrix[i, j] += result['win_rate']
            win_matrix[j, i] += 1 - result['win_rate']

    strengths = bradley_terry_model(win_matrix)
    elo_ratings = 400 * np.log10(strengths / np.min(strengths))

    rankings = pd.DataFrame({
        'Model': models,
        'Elo Rating': elo_ratings
    })
    rankings = rankings.sort_values('Elo Rating', ascending=False).reset_index(drop=True)
    rankings.index += 1
    return rankings

def main():
    os.makedirs(RANDOMIZED_PATH, exist_ok=True)
    os.makedirs(TIERED_PATH, exist_ok=True)
    
    # Prepare randomized reference outputs
    randomized_reference_file = os.path.join(RANDOMIZED_PATH, "randomized_reference_outputs.json")
    prepare_randomized_reference_outputs(all_models, NUM_QUESTIONS, randomized_reference_file)

    # Prepare tiered reference outputs
    tiered_output_dir = os.path.join(TIERED_PATH, "tiered_references")
    os.makedirs(tiered_output_dir, exist_ok=True)
    tiered_reference_files = prepare_tiered_reference_outputs(tiered_models, NUM_QUESTIONS, tiered_output_dir)

    all_results = []

    # Run evaluations for each test model
    for test_model in test_models:
        # Run randomized evaluation
        random_results, random_avg_win_rates = run_randomized_evaluation(test_model, randomized_reference_file)
        save_results(random_results, random_avg_win_rates, test_model, "randomized")
        all_results.extend(random_results)
        
        # Run tiered evaluation
        tiered_results, tiered_avg_win_rates = run_tiered_evaluation(test_model, tiered_reference_files)
        save_results(tiered_results, tiered_avg_win_rates, test_model, "tiered")
        all_results.extend(tiered_results)

    # Compute Elo rankings
    elo_rankings = compute_elo_rankings(all_results)
    
    # Save Elo rankings
    elo_rankings_file = os.path.join(OUT_PATH, "elo_rankings.csv")
    elo_rankings.to_csv(elo_rankings_file, index=False)
    logging.info(f"Elo rankings saved to {elo_rankings_file}")

    logging.info("All evaluations completed.")

if __name__ == "__main__":
    main()