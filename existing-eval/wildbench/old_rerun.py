import os
import json
import random
import logging
import requests
import backoff
from urllib.parse import urljoin
from typing import List, Dict, Optional

# Constants
NUM_QUESTIONS = 1024
BASE_URL = "https://raw.githubusercontent.com/allenai/WildBench/refs/heads/main/eval_results/v2.0522/pairwise.v2/"
EVAL_MODEL = "eval%3Dgpt-4-turbo-2024-04-09"
REF_MODEL = "ref%3Dgpt-4-turbo-2024-04-09"
OUT_PATH = "results/"
RANDOMIZED_PATH = os.path.join(OUT_PATH, "randomized")
TIERED_PATH = os.path.join(OUT_PATH, "tiered")

random.seed(42)

all_models = [
    'Hermes-2-Theta-Llama-3-8B',
    'Llama-2-70b-chat-hf',
    'Llama-2-7b-chat-hf',
    'Llama-3-8B-Magpie-Align-v0.1',
    'Llama-3-Instruct-8B-SimPO-ExPO',
    'Llama-3-Instruct-8B-SimPO',
    'Meta-Llama-3-70B-Instruct',
    'Meta-Llama-3-8B-Instruct',
    'Mistral-7B-Instruct-v0.2',
    'Mixtral-8x7B-Instruct-v0.1',
    'Nous-Hermes-2-Mixtral-8x7B-DPO',
    'Phi-3-medium-128k-instruct',
    'Phi-3-mini-128k-instruct',
    'Qwen1.5-72B-Chat-greedy',
    'Qwen1.5-72B-Chat',
    'Qwen1.5-7B-Chat@together',
    'Qwen2-72B-Instruct',
    'SELM-Llama-3-8B-Instruct-iter-3',
    'SELM-Zephyr-7B-iter-3',
    'Starling-LM-7B-beta-ExPO',
    'Starling-LM-7B-beta',
    'Yi-1.5-34B-Chat',
    'Yi-1.5-6B-Chat',
    'Yi-1.5-9B-Chat',
    'claude-3-5-sonnet-20240620',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'command-r-plus',
    'command-r',
    'dbrx-instruct@together',
    'deepseek-coder-v2',
    'deepseekv2-chat',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemma-2b-it',
    'gemma-7b-it',
    'glm-4-9b-chat',
    'gpt-3.5-turbo-0125',
    'gpt-4-0125-preview',
    'gpt-4-turbo-2024-04-09',
    'gpt-4o-2024-05-13',
    'mistral-large-2402',
    'nemotron-4-340b-instruct',
    'neo_7b_instruct_v0.1-ExPO',
    'neo_7b_instruct_v0.1',
    'reka-core-20240501',
    'reka-edge',
    'reka-flash-20240226',
    'tulu-2-dpo-70b',
    'yi-large-preview',
    'yi-large'
]

test_models = [
    'SELM-Zephyr-7B-iter-3',
    'Qwen1.5-7B-Chat@together',
    'Yi-1.5-34B-Chat',
    'Mistral-7B-Instruct-v0.2',
    'Llama-2-7b-chat-hf',
    'Yi-1.5-9B-Chat',
    'Yi-1.5-6B-Chat',
    'glm-4-9b-chat'
]

tiered_models = [
    "gpt-4-turbo-2024-04-09",
    "mistral-large-2402",
    "Llama-3-Instruct-8B-SimPO",
    "neo_7b_instruct_v0.1-ExPO"
]

all_models = [model for model in all_models if model not in test_models]
all_models = [model.replace('.json', '') for model in all_models]
test_models = [model.replace('.json', '') for model in test_models]
tiered_models = [model.replace('.json', '') for model in tiered_models]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_url(model: str) -> str:
    path = f"{EVAL_MODEL}/{REF_MODEL}/{model}.json"
    url = urljoin(BASE_URL, path)
    return url
    # return unquote(url)

@backoff.on_exception(backoff.expo, 
                     (requests.exceptions.RequestException, 
                      requests.exceptions.ConnectionError),
                     max_tries=5)
def fetch_url(url: str) -> Optional[requests.Response]:
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {str(e)}")
        return None

def load_model_outputs(model: str) -> List[Dict]:
    model_output_url = get_model_url(model)
    logging.info(f"Requesting URL: {model_output_url}")
    
    response = fetch_url(model_output_url)
    if not response:
        return []
    
    try:
        json_data = response.json()
        
        if json_data and isinstance(json_data, list) and len(json_data) > 0:
            first_item = json_data[0]
            # print("="*25)
            print(f"Finished with {model}")
            # print("="*25)
            if 'model_outputs' in first_item:
                logging.debug(f"Available model keys: {first_item['model_outputs'].keys()}")
        
        return json_data

    except Exception as err:
        logging.warning(f"Error processing JSON for {model}: {err}")
        return []

def find_model_key(available_keys, model_name):
    for key in available_keys:
        if model_name.lower() in key.lower():
            return key
    return None

def prepare_randomized_reference_outputs(all_models: List[str], num_questions: int, output_file: str) -> None:
    randomized_outputs = []
    selected_questions = random.sample(range(NUM_QUESTIONS), num_questions)
    
    for i in selected_questions:
        random_model = random.choice(all_models)
        model_outputs = load_model_outputs(random_model)
        
        if not model_outputs or i >= len(model_outputs):
            logging.warning(f"Skipping question {i} due to missing data for model {random_model}")
            continue
            
        try:
            available_keys = model_outputs[i]['model_outputs'].keys()
            logging.debug(f"Available model keys for question {i}: {available_keys}")
            
            model_key = find_model_key(available_keys, random_model)
            output_text = model_outputs[i]['model_outputs'][model_key]
            
            output_dict = {
                'output': output_text,
                'generator': random_model
            }
            randomized_outputs.append(output_dict)
            
        except KeyError as e:
            logging.warning(f"KeyError accessing data for model {random_model}: {e}")
            logging.debug(f"Available keys: {model_outputs[i]['model_outputs'].keys() if model_outputs and i < len(model_outputs) else 'None'}")
            continue

    with open(output_file, 'w') as f:
        json.dump(randomized_outputs, f, indent=2)
    logging.info(f"Randomized reference outputs saved to {output_file}")

def prepare_tiered_reference_outputs(tiered_models: List[str], num_questions: int, output_dir: str) -> List[str]:
    tier_files = []
    selected_questions = random.sample(range(NUM_QUESTIONS), num_questions)
    
    for i, model in enumerate(tiered_models):
        outputs = []
        model_outputs = load_model_outputs(model)
        
        for j in selected_questions:
            if not model_outputs or j >= len(model_outputs):
                logging.warning(f"Skipping question {j} due to missing data for model {model}")
                continue
                
            try:
                available_keys = model_outputs[j]['model_outputs'].keys()
                logging.debug(f"Available model keys for question {j}: {available_keys}")
                
                model_key = find_model_key(available_keys, model)
                output_text = model_outputs[j]['model_outputs'][model_key]
                
                output_dict = {
                    'output': output_text,
                    'generator': model
                }
                outputs.append(output_dict)
                
            except KeyError as e:
                logging.warning(f"KeyError accessing data for model {model}: {e}")
                logging.debug(f"Available keys: {model_outputs[j]['model_outputs'].keys() if model_outputs and j < len(model_outputs) else 'None'}")
                continue
        
        output_file = os.path.join(output_dir, f"tiered_reference_outputs_{i+1}.json")
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        tier_files.append(output_file)
        logging.info(f"Tiered reference outputs for tier {i+1} saved to {output_file}")
    
    return tier_files

def prepare_test_model_outputs(test_models: List[str], num_questions: int) -> None:
    selected_questions = random.sample(range(NUM_QUESTIONS), num_questions)
    
    for model in test_models:
        model_outputs = load_model_outputs(model)
        
        randomized_model_dir = os.path.join(RANDOMIZED_PATH, model)
        tiered_model_dir = os.path.join(TIERED_PATH, model)
        os.makedirs(randomized_model_dir, exist_ok=True)
        os.makedirs(tiered_model_dir, exist_ok=True)

        for tier_num in range(1, 5):
            os.makedirs(os.path.join(tiered_model_dir, f'tier_{tier_num}'), exist_ok=True)

        try:
            selected_outputs = []
            for i in selected_questions:
                if not model_outputs or i >= len(model_outputs):
                    logging.warning(f"Skipping question {i} due to missing data for model {model}")
                    continue
                    
                available_keys = model_outputs[i]['model_outputs'].keys()
                logging.debug(f"Available model keys for question {i}: {available_keys}")
                
                model_key = find_model_key(available_keys, model)
                output_text = model_outputs[i]['model_outputs'][model_key]
                
                output_dict = {
                    'output': output_text,
                    'generator': model
                }
                selected_outputs.append(output_dict)
                
        except KeyError as e:
            logging.warning(f"KeyError accessing data for model {model}: {e}")
            logging.debug(f"Available keys: {model_outputs[i]['model_outputs'].keys() if model_outputs and i < len(model_outputs) else 'None'}")
            continue

        randomized_output_file = os.path.join(randomized_model_dir, "model_outputs.json")
        with open(randomized_output_file, 'w') as f:
            json.dump(selected_outputs, f, indent=2)

        tiered_output_file = os.path.join(tiered_model_dir, "model_outputs.json")
        with open(tiered_output_file, 'w') as f:
            json.dump(selected_outputs, f, indent=2)

        logging.info(f"Prepared output files for {model}")

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create necessary directories
    os.makedirs(RANDOMIZED_PATH, exist_ok=True)
    os.makedirs(TIERED_PATH, exist_ok=True)

    # Prepare randomized reference outputs
    randomized_reference_file = os.path.join(RANDOMIZED_PATH, "randomized_reference_outputs.json")
    prepare_randomized_reference_outputs(test_models + tiered_models, NUM_QUESTIONS // 2, randomized_reference_file)

    # Prepare tiered reference outputs
    tiered_output_dir = os.path.join(TIERED_PATH, "tiered_references")
    os.makedirs(tiered_output_dir, exist_ok=True)
    prepare_tiered_reference_outputs(tiered_models, NUM_QUESTIONS // len(tiered_models), tiered_output_dir)

    # Prepare test model outputs
    prepare_test_model_outputs(test_models, NUM_QUESTIONS // len(test_models))

if __name__ == "__main__":
    main()