import os
import json
import random
import logging
from typing import List, Dict

# Constants
NUM_QUESTIONS = 805  # Updated from 200 to 805
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/final-run/"
RANDOMIZED_PATH = os.path.join(OUT_PATH, "randomized")
TIERED_PATH = os.path.join(OUT_PATH, "tiered")

# Set a fixed random seed for consistency
random.seed(42)

all_models = [
    "GPT-4o-2024-05-13",  # GPT-4 Omni (05/13)
    "gpt-4-turbo-2024-04-09",  # GPT-4 Turbo (04/09)
    "Yi-34B-Chat",  # Yi 34B Chat
    "gpt4_1106_preview",  # GPT-4 Preview (11/06)
    "claude-3-opus-20240229",  # Claude 3 Opus (02/29)
    "Llama-3-Instruct-8B-SimPO-ExPO",  # Llama 3 8B Instruct
    "Llama-3-Instruct-8B-SimPO",  # Llama 3 8B Instruct
    "Meta-Llama-3-70B-Instruct",  # Llama 3 70B Instruct
    "Meta-Llama-3-8B-Instruct",  # Llama 3 8B Instruct
    "Qwen1.5-72B-Chat",  # Qwen1.5 72B Chat
    "gpt4_0314",  # GPT-4 (03/14)
    "claude-3-sonnet-20240229",  # Claude 3 Sonnet (02/29)
    "Mixtral-8x22B-Instruct-v0.1",  # Mixtral 8x22B v0.1
    "gpt4_0613",  # GPT-4 (06/13)
    "Contextual-KTO-Mistral-PairRM",  # Contextual AI (KTO-Mistral-PairRM)
    "Mistral-7B-Instruct-v0.2",  # Mistral 7B v0.2
    "OpenHermes-2.5-Mistral-7B",  # OpenHermes-2.5-Mistral (7B)
    "Qwen1.5-7B-Chat",  # Qwen1.5 7B Chat
    "TempNet-LLaMA2-Chat-70B-v0.1",  # LLaMA2 Chat 70B
    "TempNet-LLaMA2-Chat-13B-v0.1",  # LLaMA2 Chat 13B
    "TempNet-LLaMA2-Chat-7B-v0.1",  # LLaMA2 Chat 7B
    "alpaca-7b",  # Alpaca 7B
    "alpaca-farm-ppo-human",  # Alpaca Farm PPO Human 7B
    "alpaca-farm-ppo-sim-gpt4-20k",  # Alpaca Farm PPO Sim (GPT-4) 7B
    "falcon-40b-instruct",  # Falcon 40B Instruct
    "falcon-7b-instruct",  # Falcon 7B Instruct
]

# Test models (subset of all_models, close in performance)
test_models = [
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

# Tiered models (subset of all_models, not overlapping with test_models)
tiered_models = [
    "gpt-4-turbo-2024-04-09",
    "vicuna-33b-v1.3",
    "llama-2-7b-chat-hf",
    "oasst-sft-pythia-12b"
]

all_models = [model for model in all_models if model not in test_models and model not in tiered_models]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_outputs(model: str) -> List[Dict]:
    model_output_file = os.path.join(RESULTS_PATH, f"{model}/model_outputs.json")
    try:
        with open(model_output_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Output file for {model} not found. Skipping.")
        return []

# def prepare_randomized_reference_outputs(all_models: List[str], num_questions: int, output_file: str) -> None:
#     randomized_outputs = []
#     selected_questions = random.sample(range(NUM_QUESTIONS), num_questions)
#     for i in selected_questions:
#         random_model = random.choice(all_models)
#         model_outputs = load_model_outputs(random_model)
#         if model_outputs and i < len(model_outputs):
#             output = model_outputs[i]
#             output['generator'] = random_model
#             randomized_outputs.append(output)
#         else:
#             logging.warning(f"Skipping question {i} due to missing data for model {random_model}")

#     with open(output_file, 'w') as f:
#         json.dump(randomized_outputs, f, indent=2)
#     logging.info(f"Randomized reference outputs saved to {output_file}")

def prepare_tiered_reference_outputs(tiered_models: List[str], num_questions: int, output_dir: str) -> List[str]:
    tier_files = []
    # Use all questions instead of a random sample
    selected_questions = range(NUM_QUESTIONS)  # Changed to include all questions
    for i, model in enumerate(tiered_models):
        outputs = []
        model_outputs = load_model_outputs(model)
        for j in selected_questions:
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

def prepare_test_model_outputs(test_models: List[str], num_questions: int) -> None:
    # Use all questions instead of a random sample
    selected_questions = range(NUM_QUESTIONS)  # Changed to include all questions
    for model in test_models:
        model_outputs = load_model_outputs(model)
        
        # Create directories for randomized and tiered evaluations
        randomized_model_dir = os.path.join(RANDOMIZED_PATH, model)
        tiered_model_dir = os.path.join(TIERED_PATH, model)
        os.makedirs(randomized_model_dir, exist_ok=True)
        os.makedirs(tiered_model_dir, exist_ok=True)

        # Create tier 1-4 folders in the tiered directory
        for tier_num in range(1, 5):
            os.makedirs(os.path.join(tiered_model_dir, f'tier_{tier_num}'), exist_ok=True)

        # Select all questions for each model
        selected_outputs = [model_outputs[i] for i in selected_questions if i < len(model_outputs)]

        # Prepare output file for randomized evaluation
        randomized_output_file = os.path.join(randomized_model_dir, "model_outputs.json")
        with open(randomized_output_file, 'w') as f:
            json.dump(selected_outputs, f, indent=2)

        # Prepare output file for tiered evaluation
        tiered_output_file = os.path.join(tiered_model_dir, "model_outputs.json")
        with open(tiered_output_file, 'w') as f:
            json.dump(selected_outputs, f, indent=2)

        logging.info(f"Prepared output files for {model}")

def main():
    # Create necessary directories
    os.makedirs(RANDOMIZED_PATH, exist_ok=True)
    os.makedirs(TIERED_PATH, exist_ok=True)

    # Prepare randomized reference outputs
    # randomized_reference_file = os.path.join(RANDOMIZED_PATH, "randomized_reference_outputs.json")
    # prepare_randomized_reference_outputs(all_models, NUM_QUESTIONS, randomized_reference_file)

    # Prepare tiered reference outputs
    tiered_output_dir = os.path.join(TIERED_PATH, "tiered_references")
    os.makedirs(tiered_output_dir, exist_ok=True)
    tiered_reference_files = prepare_tiered_reference_outputs(tiered_models, NUM_QUESTIONS, tiered_output_dir)

    # Prepare test model outputs
    prepare_test_model_outputs(test_models, NUM_QUESTIONS)

    logging.info("All directories created and model outputs prepared.")

    # Generate and print alpaca_eval commands
    print("\nAlpaca Eval Commands:")
    for test_model in test_models:
        print(f"\n--- Commands for {test_model} ---")
        
        # # Command for randomized output
        # print("\nRandomized Evaluation:")
        # randomized_output_path = os.path.join(OUT_PATH, "randomized", test_model)
        # print(f"alpaca_eval --model_outputs {os.path.join(RANDOMIZED_PATH, test_model, 'model_outputs.json')} --reference_outputs {randomized_reference_file} --annotators_config alpaca_eval_gpt4_turbo_fn --output_path {randomized_output_path}")
        
        # Commands for tiered models
        print("\nTiered Evaluations:")
        for i, tiered_model in enumerate(tiered_models, start=1):
            tiered_output_path = os.path.join(OUT_PATH, "tiered", test_model, f"tier_{i}")
            print(f"\nTier {i} ({tiered_model}):")
            print(f"alpaca_eval --model_outputs {os.path.join(TIERED_PATH, test_model, 'model_outputs.json')} --reference_outputs {os.path.join(TIERED_PATH, 'tiered_references', f'tiered_reference_outputs_{i}.json')} --annotators_config alpaca_eval_gpt4_turbo_fn --output_path {tiered_output_path}")

    logging.info("All directories created, model outputs prepared, and alpaca_eval commands generated.")

if __name__ == "__main__":
    main()
