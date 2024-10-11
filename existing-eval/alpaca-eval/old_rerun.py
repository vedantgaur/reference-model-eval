import random
import os
import json
from alpaca_eval import evaluate
from collections import defaultdict
import datasets
from typing import Literal
import pandas as pd
from alpaca_eval.annotators import BaseAnnotator

# Constants
NUM_QUESTIONS = 850  
THRESHOLD = 0.20 
LEADERBOARD_FILE = 'results/leaderboard.json'

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

# Define our tiered reference models (adjust as needed)
tiered_models = [
    "gpt-4",  # Really good
    "claude-v1",  # Fairly good
    "gpt-3.5-turbo",  # Average
    "vicuna-13b",  # Somewhat bad
    "stablelm-tuned-alpha-7b"  # Bottom
]

# Test models (you can adjust this list as needed)
test_models = [
                'stablelm-tuned-alpha-7b', 
                'vicuna-13b', 
                'alpaca-13b', 
                'wizardlm-13b', 
                'gpt-3.5-turbo', 
                'mpt-7b-chat', 
                'gpt4all-13b-snoozy', 
                'guanaco-33b', 
                'RWKV-4-Raven-14B', 
                'koala-13b', 
                'vicuna-7b', 
                'dolly-v2-12b', 
                'gpt-4', 
                'oasst-pythia-12b', 
                'fastchat-t5-3b', 
                'claude-v1', 
                'claude-instant-v1', 
                'llama-13b', 
                'chatglm-6b', 
                'palm-2'
            ]

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_model_outputs(model_name):
    file_path = f"~/Projects/alpaca_eval/results/{model_name}/model_outputs.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def run_evaluation(test_model, reference_models, questions, type: Literal["random", "tiered"]):
    results = defaultdict(list)

    current_tier = 0
    rounds_per_tier = 50  
    current_tier_rounds = 0

    # Load pre-existing model outputs
    test_model_outputs = load_model_outputs(test_model)

    # Create a GPT4Annotator instance
    gpt4_annotator = BaseAnnotator(model="gpt-4-turbo-2024-04-09")  # Use the latest GPT-4 model

    curr_round = 1
    for question in questions:
        print(f"Evaluating Round {curr_round}")
        curr_round += 1
        if type == "tiered":
            reference_model = reference_models[current_tier]
        elif type == "random":
            reference_model = random.choice(reference_models)
        
        # Use pre-existing outputs if available, otherwise compute
        if test_model_outputs:
            test_output = next((item for item in test_model_outputs if item['instruction'] == question['instruction']), None)
            if test_output:
                model_output = [test_output]
            else:
                model_output = test_model
        else:
            model_output = test_model

        # Use GPT-4 for annotation
        df_leaderboard, annotations = evaluate(
            model_outputs=model_output,
            reference_outputs=reference_model,
            annotators_config=gpt4_annotator,
            questions=[question],
            is_return_instead_of_print=True,
            output_path=f"results/{test_model}_{type}",
            is_cache_leaderboard=False,  # Disable caching
            max_instances=1  # Evaluate one question at a time
        )

        # Store the results
        results[reference_model].append({
            'df_leaderboard': df_leaderboard,
            'annotations': annotations
        })

        if type == "tiered":
            current_tier_rounds += 1
            if current_tier_rounds >= rounds_per_tier:
                winrate = df_leaderboard.loc[test_model, 'win_rate'] if test_model in df_leaderboard.index else 0
                if winrate < THRESHOLD and current_tier < len(reference_models) - 1:
                    current_tier += 1
                current_tier_rounds = 0
    
    return results

def aggregate_results(results):
    aggregated = {}
    for model, evals in results.items():
        df_combined = pd.concat([eval_['df_leaderboard'] for eval_ in evals])
        aggregated[model] = {
            'win_rate': df_combined['win_rate'].mean(),
            'standard_error': df_combined['standard_error'].mean(),
            'total_comparisons': df_combined['n_total'].sum(),
            'avg_length': df_combined['avg_length'].mean()
        }
    return aggregated

def update_leaderboard(aggregated_results):
    leaderboard = load_leaderboard()
    for model, metrics in aggregated_results.items():
        leaderboard[model] = leaderboard.get(model, {})
        leaderboard[model]['win_rate'] = metrics['win_rate']
        leaderboard[model]['loss_rate'] = metrics['loss_rate']
        leaderboard[model]['tie_rate'] = metrics['tie_rate']
        leaderboard[model]['total_comparisons'] = metrics['total_comparisons']
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)

def save_results(results, aggregated, test_model, method):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Convert DataFrame to dictionary for JSON serialization
    def convert_df_to_dict(df):
        if isinstance(df, pd.DataFrame):
            return df.to_dict(orient='records')
        return df

    # Convert results to JSON-serializable format
    json_results = {
        model: [
            {
                'df_leaderboard': convert_df_to_dict(item['df_leaderboard']),
                'annotations': item['annotations']
            }
            for item in evals
        ]
        for model, evals in results.items()
    }
    
    # Save detailed results
    with open(f'results/{test_model}_{method}_detailed.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save aggregated results
    with open(f'results/{test_model}_{method}_aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Update the leaderboard
    update_leaderboard(aggregated)

def main():
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    examples = [dict(example) for example in eval_set]
    
    for test_model in test_models:
        # Run with random reference models
        random_results = run_evaluation(test_model, all_models, examples, "random")
        random_aggregated = aggregate_results(random_results)
        save_results(random_results, random_aggregated, test_model, "random")
        
        # Run with tiered reference models
        tiered_results = run_evaluation(test_model, tiered_models, examples, "tiered")
        tiered_aggregated = aggregate_results(tiered_results)
        save_results(tiered_results, tiered_aggregated, test_model, "tiered")

if __name__ == "__main__":
    main()
