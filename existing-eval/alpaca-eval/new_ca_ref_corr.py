import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def load_data(alpaca_file, arena_file):
    alpaca_data = pd.read_csv(alpaca_file)
    arena_data = pd.read_csv(arena_file)
    return alpaca_data, arena_data

def divide_into_percentiles(scores):
    return pd.qcut(scores, q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])

def calculate_correlations(merged_data):
    percentile_ranges = ['0-25%', '25-50%', '50-75%', '75-100%']
    correlations = {range_label: [] for range_label in percentile_ranges}
    
    for _, row in merged_data.iterrows():
        model = row['Model']
        model_percentile = row['Percentile_Range']
        other_models = merged_data[merged_data['Model'] != model]
        
        for range_label in percentile_ranges:
            range_models = other_models[other_models['Percentile_Range'] == range_label]
            if len(range_models) > 1:
                corr, _ = pearsonr(range_models['Win_Rate'], range_models['Arena Score'])
                correlations[range_label].append(corr)
            else:
                correlations[range_label].append(np.nan)
    
    return pd.DataFrame(correlations, index=merged_data['Model'])

def main(alpaca_file, arena_file):
    alpaca_data, arena_data = load_data(alpaca_file, arena_file)
    
    merged_data = pd.merge(alpaca_data, arena_data, on='Model', how='inner')
    merged_data = merged_data.sort_values('Win_Rate', ascending=False).reset_index(drop=True)
    
    merged_data['Percentile_Range'] = divide_into_percentiles(merged_data['Win_Rate'])
    
    correlations = calculate_correlations(merged_data)
    
    output_matrix = pd.concat([
        merged_data['Model'],
        correlations.round(4),
        merged_data['Percentile_Range']
    ], axis=1)
    
    print(output_matrix)
    print("\nMean Correlations:")
    print(correlations.mean().round(4))
    
    return output_matrix

if __name__ == "__main__":
    alpaca_file = "data/alpaca_eval_scores.csv"
    arena_file = "~/Projects/llm-eval/chatbot-arena/data/chatbot_arena_scores.csv"
    result = main(alpaca_file, arena_file)
    result.to_csv("output_matrix.csv", index=False, float_format='%.4f')