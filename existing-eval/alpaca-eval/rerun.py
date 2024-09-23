import random
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Assuming these are imported from the AlpacaEval codebase
from alpaca_eval import load_dataset, generate_response, evaluate_preference

class EnhancedMultiReferenceAlpacaEval:
    def __init__(self, models: Dict[str, Dict], dataset: str, num_iterations: int = 5):
        self.models = models
        self.dataset = load_dataset(dataset)
        self.num_iterations = num_iterations

    def generate_all_responses(self) -> Dict[str, List[str]]:
        all_responses = {model: [] for model in self.models}
        for example in tqdm(self.dataset, desc="Generating responses for all models"):
            for model in self.models:
                response = generate_response(model, example['instruction'])
                all_responses[model].append(response)
        return all_responses

    def evaluate_model_pair(self, model1: str, model2: str, responses: Dict[str, List[str]]) -> float:
        score = 0
        for i, example in enumerate(self.dataset):
            preference = evaluate_preference(responses[model1][i], responses[model2][i], example['instruction'])
            score += preference
        return score / len(self.dataset)

    def run_evaluation(self) -> pd.DataFrame:
        print(f"Evaluating {len(self.models)} models")
        all_responses = self.generate_all_responses()
        
        results = []
        for _ in tqdm(range(self.num_iterations), desc="Running evaluation iterations"):
            iteration_results = []
            for model in self.models:
                model_score = 0
                for ref_model in self.models:
                    if model != ref_model:
                        model_score += self.evaluate_model_pair(model, ref_model, all_responses)
                avg_score = model_score / (len(self.models) - 1)
                iteration_results.append((model, avg_score))
            results.extend(iteration_results)

        df = pd.DataFrame(results, columns=['Model', 'Score'])
        final_results = df.groupby('Model')['Score'].agg(['mean', 'std']).reset_index()
        final_results = final_results.sort_values('mean', ascending=False).reset_index(drop=True)
        final_results['Rank'] = final_results.index + 1
        final_results = final_results[['Rank', 'Model', 'mean', 'std']]
        final_results.columns = ['Rank', 'Model', 'Mean Score', 'Std Dev']
        
        print("\nFinal Leaderboard:")
        print(final_results.to_string(index=False))
        return final_results

# Usage example
if __name__ == "__main__":
    models = {
        "gpt-4": {"type": "api", "provider": "openai"},
        "gpt-3.5-turbo": {"type": "api", "provider": "openai"},
        "claude-2": {"type": "api", "provider": "anthropic"},
        "claude-instant-1": {"type": "api", "provider": "anthropic"},
        "palm-2": {"type": "api", "provider": "google"},
        "llama-2-70b": {"type": "local", "path": ""},
        "falcon-40b": {"type": "local", "path": ""},
        "vicuna-13b": {"type": "local", "path": ""},
        "alpaca-7b": {"type": "local", "path": ""},
        "flan-t5-xxl": {"type": "huggingface", "model_id": ""},
        "opt-66b": {"type": "local", "path": ""},
        "bloom-176b": {"type": "huggingface", "model_id": "bigscience/bloom"},
        "pythia-12b": {"type": "local", "path": ""},
        "dolly-v2-12b": {"type": "local", "path": ""},
        "stablelm-tuned-alpha-7b": {"type": "huggingface", "model_id": "stabilityai/stablelm-tuned-alpha-7b"},
    }
    
    dataset = "alpaca_eval_dataset"
    
    evaluator = EnhancedMultiReferenceAlpacaEval(models, dataset)
    leaderboard = evaluator.run_evaluation()

    leaderboard.to_csv("alpaca_eval_leaderboard.csv", index=False)