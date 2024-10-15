import os
import json
import random
import logging
from typing import List, Dict

# Constants
NUM_QUESTIONS = 200
RESULTS_PATH = '/Users/vedantgaur/Projects/alpaca_eval/results'
OUT_PATH = "results/new-run/"
TIERED_PATH = os.path.join(OUT_PATH, "tiered")

random.seed(42)

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

def prepare_gpt4_reference_outputs(num_questions: int, output_dir: str) -> str:
    gpt4_model = "gpt4_1106_preview"
    outputs = []
    selected_questions = random.sample(range(NUM_QUESTIONS), num_questions)
    model_outputs = load_model_outputs(gpt4_model)
    for j in selected_questions:
        if model_outputs and j < len(model_outputs):
            output = model_outputs[j]
            output['generator'] = gpt4_model
            outputs.append(output)
        else:
            logging.warning(f"Skipping question {j} due to missing data for model {gpt4_model}")
    
    output_file = os.path.join(output_dir, "tiered_reference_outputs_1.json")
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    logging.info(f"GPT-4 reference outputs saved to {output_file}")
    
    return output_file

def main():
    # Prepare GPT-4 reference outputs for tier_1
    tiered_output_dir = os.path.join(TIERED_PATH, "tiered_references")
    os.makedirs(tiered_output_dir, exist_ok=True)
    gpt4_reference_file = prepare_gpt4_reference_outputs(NUM_QUESTIONS, tiered_output_dir)

    for test_model in test_models:
        print(f"\n--- Commands for {test_model} ---")
        # Command for tier_1 (GPT-4) evaluation
        print("\nTier 1 Evaluation (GPT-4):")
        tiered_output_path = os.path.join(OUT_PATH, "tiered", test_model, "tier_1")
        print(f"alpaca_eval --model_outputs {os.path.join(TIERED_PATH, test_model, 'model_outputs.json')} --reference_outputs {gpt4_reference_file} --annotators_config alpaca_eval_gpt4_turbo_fn --output_path {tiered_output_path}")

    logging.info("All directories created, model outputs prepared, and alpaca_eval commands generated.")

if __name__ == "__main__":
    main()