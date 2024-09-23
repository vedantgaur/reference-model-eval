import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from wildbench import load_dataset, generate_response, evaluate_wb_reward, evaluate_wb_score

class ConstantReferenceWildBench:
    def __init__(self, constant_reference: str, models_to_evaluate: List[str], dataset: str):
        self.constant_reference = constant_reference
        self.models_to_evaluate = models_to_evaluate
        self.dataset = load_dataset(dataset)

    def generate_reference_responses(self) -> List[str]:
        print(f"Generating responses for constant reference model: {self.constant_reference}")
        reference_responses = []
        for example in tqdm(self.dataset):
            response = generate_response(self.constant_reference, example['instruction'])
            reference_responses.append(response)
        return reference_responses

    def evaluate_models(self, reference_responses: List[str]) -> pd.DataFrame:
        results = []
        for model in tqdm(self.models_to_evaluate, desc="Evaluating models"):
            wb_reward_scores = []
            wb_scores = []
            for i, example in enumerate(self.dataset):
                model_response = generate_response(model, example['instruction'])
                reference_response = reference_responses[i]
                
                wb_reward = evaluate_wb_reward(model_response, reference_response, example['instruction'])
                wb_score = evaluate_wb_score(model_response, example['instruction'])
                
                wb_reward_scores.append(wb_reward)
                wb_scores.append(wb_score)
            
            avg_wb_reward = np.mean(wb_reward_scores)
            avg_wb_score = np.mean(wb_scores)
            results.append({
                'Model': model,
                'WB-Reward': avg_wb_reward,
                'WB-Score': avg_wb_score
            })
        
        return pd.DataFrame(results)

    def run_evaluation(self) -> pd.DataFrame:
        print(f"Evaluating {len(self.models_to_evaluate)} models against constant reference: {self.constant_reference}")
        reference_responses = self.generate_reference_responses()
        results = self.evaluate_models(reference_responses)
        
        results = results.sort_values('WB-Reward', ascending=False).reset_index(drop=True)
        results['Rank'] = results.index + 1
        results = results[['Rank', 'Model', 'WB-Reward', 'WB-Score']]
        
        print("\nFinal Leaderboard:")
        print(results.to_string(index=False))
        return results

# Usage example
if __name__ == "__main__":
    constant_reference = "gpt-4-turbo-0429"
    models_to_evaluate = [
        "claude-3-haiku",
        "llama-2-70b-chat",
        "gpt-3.5-turbo",
        "palm-2",
        "claude-2",
        "vicuna-13b",
        "falcon-40b",
        "alpaca-7b",
        "flan-t5-xxl",
        "opt-66b",
        "bloom-176b",
        "pythia-12b",
        "dolly-v2-12b",
        "stablelm-tuned-alpha-7b",
    ]
    dataset = "wildbench_dataset"
    
    evaluator = ConstantReferenceWildBench(constant_reference, models_to_evaluate, dataset)
    leaderboard = evaluator.run_evaluation()

    leaderboard.to_csv("wildbench_constant_reference_leaderboard.csv", index=False)