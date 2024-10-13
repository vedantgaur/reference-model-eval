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
# NUM_QUESTIONS = 850
THRESHOLD = 0.20
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/"
LEADERBOARD_FILE = os.path.join(RESULTS_PATH, 'leaderboard.json')
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
    "Storm-7B-num-beams-10",
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
    "claude-instant-1.2",  # Fairly good
    "gpt-3.5-turbo-1106",  # Average
    "vicuna-13b",  # Somewhat bad
    "falcon-7b-instruct"  # Bottom
]

# Test models 
test_models = [
                'vicuna-13b', 
                # 'alpaca-13b', 
                'wizardlm-13b', 
                'gpt-3.5-turbo-1106', 
                # 'mpt-7b-chat', 
                'gpt4_1106_preview', 
                'guanaco-33b', 
                # 'RWKV-4-Raven-14B', 
                # 'koala-13b', 
                'vicuna-7b', 
                # 'dolly-v2-12b', 
                "gemma-7b-it",
                'gpt-4o-2024-05-13', 
                'oasst-sft-pythia-12b', 
                # 'fastchat-t5-3b', 
                'claude', 
                'claude-instant-1.2', 
                'llama-2-13b-chat-hf', 
                'chatglm2-6b', 
                # 'palm-2'
            ]

def prepare_all_model_outputs(test_models: List[str], questions: List[Dict], output_file: str) -> None:
    """
    Prepare a single JSON file containing outputs from all test models.
    """
    all_outputs = []
    for model in test_models:
        model_output_file = os.path.join(RESULTS_PATH, f"{model}/model_outputs.json")
        try:
            with open(model_output_file, 'r') as f:
                model_outputs = json.load(f)
            for output in model_outputs:
                output['generator'] = model  # Add the model name as the generator
                all_outputs.append(output)
        except FileNotFoundError:
            logging.warning(f"Output file for {model} not found. Skipping.")

    # Write all outputs to a single file in OUT_PATH
    with open(output_file, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    logging.info(f"All model outputs saved to {output_file}")

def prepare_reference_outputs(reference_models: List[str], questions: List[Dict], output_file: str) -> None:
    """
    Prepare a single JSON file containing outputs from reference models.
    """
    reference_outputs = []
    for model in reference_models:
        model_output_file = os.path.join(RESULTS_PATH, f"{model}/model_outputs.json")
        try:
            with open(model_output_file, 'r') as f:
                model_outputs = json.load(f)
            reference_outputs.extend(model_outputs)
        except FileNotFoundError:
            logging.warning(f"Output file for reference model {model} not found. Skipping.")

    # Write reference outputs to a single file in OUT_PATH
    with open(output_file, 'w') as f:
        json.dump(reference_outputs, f, indent=2)
    logging.info(f"Reference outputs saved to {output_file}")

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

def run_alpaca_eval(model_outputs: str, reference_outputs: str, output_path: str) -> None:
    """Run alpaca_eval command using subprocess."""
    cmd = [
        "alpaca_eval",
        "--model_outputs", model_outputs,
        "--reference_outputs", reference_outputs,
        "--annotators_config", "alpaca_eval_gpt4_turbo_fn",
        "--output_path", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running alpaca_eval: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise

def get_win_rate(output_path: str) -> float:
    """Extract win rate from the evaluation results."""
    try:
        with open(os.path.join(output_path, 'leaderboard.json'), 'r') as f:
            leaderboard = json.load(f)
        return leaderboard[list(leaderboard.keys())[0]]['win_rate']
    except Exception as e:
        logging.error(f"Error extracting win rate: {e}")
        return 0.0

def run_evaluation(test_model: str, reference_models: List[str], questions: List[Dict], eval_type: str) -> Tuple[List[Dict], Dict[str, float]]:
    """Run evaluation for a test model against reference models."""
    current_tier = 0
    current_tier_rounds = 0
    results = []
    win_rates = defaultdict(list)

    for i, question in enumerate(questions):
        logging.info(f"Evaluating Round {i+1} for {test_model} ({eval_type} mode)")
        
        if eval_type == "tiered":
            reference_model = reference_models[current_tier]
        elif eval_type == "random":
            reference_model = random.choice(reference_models)
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

        # Print the required information
        print(f"Evaluation Round {i+1}:")
        print(f"a. Model being evaluated: {test_model}")
        print(f"b. Reference model: {reference_model}")
        print(f"c. Type of reference: {eval_type}")
        print(f"d. Question number: {i+1}")
        print(f"Question: {question['instruction']}")
        print("---")


        # Prepare output files for this round
        model_output_file = os.path.join(OUT_PATH, f"{test_model}_output.json")
        reference_output_file = os.path.join(OUT_PATH, f"{reference_model}_output.json")
        
        # Write model outputs to files
        with open(model_output_file, 'w') as f:
            json.dump([question], f)
        
        # For reference model, read from RESULTS_PATH but write to OUT_PATH
        reference_input_file = os.path.join(RESULTS_PATH, f"{reference_model}/model_outputs.json")
        with open(reference_input_file, 'r') as f:
            reference_data = json.load(f)
        with open(reference_output_file, 'w') as f:
            json.dump([reference_data[i]], f)

        # Run alpaca_eval
        output_path = os.path.join(OUT_PATH, f"{test_model}_{eval_type}_{i}")
        run_alpaca_eval(model_output_file, reference_output_file, output_path)

        # Get results
        win_rate = get_win_rate(output_path)
        results.append({
            'round': i,
            'test_model': test_model,
            'reference_model': reference_model,
            'win_rate': win_rate,
            'tier': current_tier if eval_type == "tiered" else None
        })
        win_rates[reference_model].append(win_rate)

        if eval_type == "tiered":
            current_tier_rounds += 1
            if current_tier_rounds >= ROUNDS_PER_TIER:
                avg_win_rate = sum(win_rates[reference_model]) / len(win_rates[reference_model])
                logging.info(f"Average win rate for tier {current_tier}: {avg_win_rate}")
                if avg_win_rate < THRESHOLD and current_tier < len(reference_models) - 1:
                    current_tier += 1
                    logging.info(f"Moving to tier {current_tier}")
                current_tier_rounds = 0
                win_rates[reference_model] = []

    avg_win_rates = {model: sum(rates) / len(rates) for model, rates in win_rates.items()}
    return results, avg_win_rates

def save_results(results: List[Dict], avg_win_rates: Dict[str, float], test_model: str, eval_type: str) -> None:
    """Save evaluation results and average win rates."""
    output_dir = os.path.join(OUT_PATH, eval_type)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{test_model}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'avg_win_rates': avg_win_rates
        }, f, indent=2)
    logging.info(f"Results saved to {output_file}")

def load_questions(file_path: str) -> List[Dict]:
    """Load questions from a JSON file."""
    with open(file_path, 'r') as f:
        questions = json.load(f)
    return questions

def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    questions = [dict(example) for example in eval_set]
    
    # Prepare paths
    all_model_outputs = os.path.join(OUT_PATH, "all_model_outputs.json")
    reference_outputs = os.path.join(OUT_PATH, "reference_outputs.json")
    leaderboard_path = os.path.join(OUT_PATH, "leaderboard.csv")
    annotators_config = "alpaca_eval_gpt4_turbo_fn"  # Or path to your custom config

    # Prepare all_model_outputs JSON
    prepare_all_model_outputs(test_models, questions, all_model_outputs)
    
    # Prepare reference_outputs JSON
    prepare_reference_outputs(tiered_models, questions, reference_outputs)

    # Run the leaderboard generation
    run_alpaca_eval_leaderboard(all_model_outputs, reference_outputs, leaderboard_path, annotators_config)

    logging.info(f"Leaderboard saved to {leaderboard_path}")
    
    for test_model in test_models:
        # Run with random reference models
        random_results, random_avg_win_rates = run_evaluation(test_model, all_models, questions, "random")
        save_results(random_results, random_avg_win_rates, test_model, "randomized")
        
        # Run with tiered reference models
        tiered_results, tiered_avg_win_rates = run_evaluation(test_model, tiered_models, questions, "tiered")
        save_results(tiered_results, tiered_avg_win_rates, test_model, "tiered")

    logging.info("All evaluations completed.")

if __name__ == "__main__":
    main()
