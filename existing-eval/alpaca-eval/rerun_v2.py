import os
import json
import random
import subprocess
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_QUESTIONS = 805
THRESHOLD = 0.20
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/new-run/"
RANDOMIZED_PATH = os.path.join(OUT_PATH, "randomized")
TIERED_PATH = os.path.join(OUT_PATH, "tiered")
ROUNDS_PER_TIER = 50

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

tiered_models = [
    "gpt4_1106_preview",  # Really good
    "Meta-Llama-3-70B-Instruct",  # Fairly good
    "gpt-3.5-turbo-1106",  # Average
    "vicuna-13b",  # Somewhat bad
    "falcon-7b-instruct"  # Bottom
]

# Test models 
test_models = [
    'vicuna-13b', 
    'wizardlm-13b', 
    'gpt-3.5-turbo-1106', 
    'gpt4_1106_preview', 
    'guanaco-33b', 
    'vicuna-7b', 
    "gemma-7b-it",
    'gpt-4o-2024-05-13', 
    'oasst-sft-pythia-12b', 
    'claude', 
    'claude-instant-1.2', 
    'llama-2-13b-chat-hf', 
    'chatglm2-6b', 
]

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
    for i in range(len(tiered_models)):
        outputs = []
        for j in range(num_questions):
            # Determine the model based on the specified conditions
            if i == 0:  # All gpt4_1106_preview
                model = "gpt4_1106_preview"
            elif i == 1:  # First 50 gpt4_1106_preview, rest Meta-Llama-3-70B-Instruct
                model = "gpt4_1106_preview" if j < 50 else "Meta-Llama-3-70B-Instruct"
            elif i == 2:  # First 50 gpt4_1106_preview, next 50 Meta-Llama-3-70B-Instruct, rest gpt-3.5-turbo-1106
                if j < 50:
                    model = "gpt4_1106_preview"
                elif j < 100:
                    model = "Meta-Llama-3-70B-Instruct"
                else:
                    model = "gpt-3.5-turbo-1106"
            elif i == 3:  # First 50 gpt4_1106_preview, next 50 Meta-Llama-3-70B-Instruct, next 50 gpt-3.5-turbo-1106, rest vicuna-13b
                if j < 50:
                    model = "gpt4_1106_preview"
                elif j < 100:
                    model = "Meta-Llama-3-70B-Instruct"
                elif j < 150:
                    model = "gpt-3.5-turbo-1106"
                else:
                    model = "vicuna-13b"
            elif i == 4:  # First 50 gpt4_1106_preview, next 50 Meta-Llama-3-70B-Instruct, next 50 gpt-3.5-turbo-1106, next 50 vicuna-13b, rest falcon-7b-instruct
                if j < 50:
                    model = "gpt4_1106_preview"
                elif j < 100:
                    model = "Meta-Llama-3-70B-Instruct"
                elif j < 150:
                    model = "gpt-3.5-turbo-1106"
                elif j < 200:
                    model = "vicuna-13b"
                else:
                    model = "falcon-7b-instruct"
            else:
                model = tiered_models[-1]  # Fallback to the lowest tier model if needed
            
            model_outputs = load_model_outputs(model)
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
        logging.info(f"Tiered reference outputs for combination {i+1} saved to {output_file}")
    
    return tier_files

def run_alpaca_eval(model_outputs: str, reference_outputs: str, output_path: str) -> None:
    cmd = [
        "alpaca_eval",
        "--model_outputs", model_outputs,
        "--reference_outputs", reference_outputs,
        "--annotators_config", "alpaca_eval_gpt4_turbo_fn",
        "--output_path", output_path
    ]
    logging.info(f"Running alpaca_eval with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"alpaca_eval completed. Stdout: {result.stdout}")
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running alpaca_eval make_leaderboard: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise

def get_win_rate(output_path: str) -> float:
    try:
        with open(os.path.join(output_path, 'leaderboard.json'), 'r') as f:
            leaderboard = json.load(f)
        return leaderboard[list(leaderboard.keys())[0]]['win_rate']
    except Exception as e:
        logging.error(f"Error extracting win rate: {e}")
        return 0.0

def run_tiered_evaluation(test_model: str, tier_files: List[str]) -> Tuple[List[Dict], Dict[str, float]]:
    results = []
    win_rates = defaultdict(list)
    model_outputs = load_model_outputs(test_model)
    model_output_file = os.path.join(TIERED_PATH, f"{test_model}_output.json")
    with open(model_output_file, 'w') as f:
        json.dump(model_outputs, f, indent=2)

    current_tier = 0
    total_questions = 0
    while current_tier < len(tier_files):
        reference_outputs = tier_files[current_tier]
        output_path = os.path.join(TIERED_PATH, f"{test_model}_tiered_{current_tier}")
        run_alpaca_eval(model_output_file, reference_outputs, output_path)

        win_rate = get_win_rate(output_path)
        results.append({
            'test_model': test_model,
            'tier': current_tier + 1,
            'win_rate': win_rate
        })
        win_rates[f"tier_{current_tier + 1}"].append(win_rate)

        total_questions += ROUNDS_PER_TIER
        if win_rate < THRESHOLD and current_tier < len(tier_files) - 1 and total_questions < NUM_QUESTIONS:
            current_tier += 1
        else:
            break

    avg_win_rates = {tier: sum(rates) / len(rates) for tier, rates in win_rates.items()}
    return results, avg_win_rates

def run_randomized_evaluation(test_model: str, reference_outputs: str) -> Tuple[List[Dict], Dict[str, float]]:
    model_outputs = load_model_outputs(test_model)
    model_output_file = os.path.join(RANDOMIZED_PATH, f"{test_model}_output.json")
    with open(model_output_file, 'w') as f:
        json.dump(model_outputs, f, indent=2)

    output_path = os.path.join(RANDOMIZED_PATH, f"{test_model}_randomized")
    run_alpaca_eval(model_output_file, reference_outputs, output_path)

    win_rate = get_win_rate(output_path)
    results = [{
        'test_model': test_model,
        'win_rate': win_rate
    }]
    avg_win_rates = {test_model: win_rate}
    return results, avg_win_rates

def save_results(results: List[Dict], avg_win_rates: Dict[str, float], test_model: str, eval_type: str) -> None:
    output_dir = os.path.join(OUT_PATH, eval_type)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{test_model}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'avg_win_rates': avg_win_rates
        }, f, indent=2)
    logging.info(f"Results saved to {output_file}")

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

    # Prepare directories for randomized and tiered evaluations
    os.makedirs(RANDOMIZED_PATH, exist_ok=True)
    os.makedirs(TIERED_PATH, exist_ok=True)

    # Prepare all_model_outputs for leaderboard creation
    all_model_outputs = {}
    for model in test_models:
        model_outputs = load_model_outputs(model)
        if model_outputs:
            all_model_outputs[model] = model_outputs

    # Run evaluations for each test model
    for test_model in test_models:
        # Run randomized evaluation
        random_results, random_avg_win_rates = run_randomized_evaluation(test_model, randomized_reference_file)
        save_results(random_results, random_avg_win_rates, test_model, "randomized")
        
        # Run tiered evaluation
        tiered_results, tiered_avg_win_rates = run_tiered_evaluation(test_model, tiered_reference_files)
        save_results(tiered_results, tiered_avg_win_rates, test_model, "tiered")

    # Create leaderboards
    randomized_leaderboard_path = os.path.join(RANDOMIZED_PATH, "leaderboard.json")
    tiered_leaderboard_path = os.path.join(TIERED_PATH, "leaderboard.json")

    run_alpaca_eval_leaderboard(
        all_model_outputs=json.dumps(all_model_outputs),
        reference_outputs=randomized_reference_file,
        leaderboard_path=randomized_leaderboard_path,
        annotators_config="alpaca_eval_gpt4_turbo_fn"
    )

    run_alpaca_eval_leaderboard(
        all_model_outputs=json.dumps(all_model_outputs),
        reference_outputs=tiered_reference_files[-1],  # Use the last tier as reference
        leaderboard_path=tiered_leaderboard_path,
        annotators_config="alpaca_eval_gpt4_turbo_fn"
    )

    logging.info("All evaluations and leaderboard creation completed.")

if __name__ == "__main__":
    main()