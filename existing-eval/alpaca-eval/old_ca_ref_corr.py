import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def load_data(alpaca_file, arena_file):
    alpaca_data = pd.read_csv(alpaca_file)
    arena_data = pd.read_csv(arena_file)
    return alpaca_data, arena_data

def normalize_model_names(name):
    return name.lower().replace(" ", "").replace("-", "").replace(".", "").replace("(", "").replace(")", "")

def find_partial_match(alpaca_models, arena_models):
    matches = {}
    for alpaca_model in alpaca_models:
        best_match = None
        best_match_length = 0
        for arena_model in arena_models:
            if alpaca_model in arena_model or arena_model in alpaca_model:
                if len(arena_model) > best_match_length:
                    best_match = arena_model
                    best_match_length = len(arena_model)
        if best_match:
            matches[alpaca_model] = best_match
    return matches

def divide_into_percentiles(scores):
    percentiles = np.percentile(scores, [25, 50, 75])
    ranges = pd.cut(scores, bins=[-np.inf] + list(percentiles) + [np.inf], 
                    labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    return ranges

def calculate_correlations(alpaca_model_score, arena_model_score, alpaca_scores, arena_scores, percentile_ranges):
    correlations = {}
    for range_label in ['0-25%', '25-50%', '50-75%', '75-100%']:
        mask = percentile_ranges == range_label
        alpaca_subset = alpaca_scores[mask]
        arena_subset = arena_scores[alpaca_subset.index]
        if len(arena_subset) > 1:
            corr, _ = pearsonr([alpaca_model_score] * len(arena_subset), arena_subset)
            correlations[range_label] = corr
        else:
            correlations[range_label] = np.nan
    return correlations

def main(alpaca_file, arena_file):
    alpaca_data, arena_data = load_data(alpaca_file, arena_file)
    
    alpaca_data['Normalized_Model'] = alpaca_data['Model'].apply(normalize_model_names)
    arena_data['Normalized_Model'] = arena_data['Model'].apply(normalize_model_names)
    
    matches = find_partial_match(alpaca_data['Normalized_Model'], arena_data['Normalized_Model'])
    
    merged_data = []
    for alpaca_model, arena_model in matches.items():
        alpaca_row = alpaca_data[alpaca_data['Normalized_Model'] == alpaca_model].iloc[0]
        arena_row = arena_data[arena_data['Normalized_Model'] == arena_model].iloc[0]
        merged_data.append({
            'Model': alpaca_row['Model'],
            'Win_Rate': alpaca_row['Win_Rate'],
            'Arena Score': arena_row['Arena Score'],
            'Normalized_Model': alpaca_model
        })
    merged_data = pd.DataFrame(merged_data)
    
    if merged_data.empty:
        print("No matching models found between Alpaca and Arena datasets.")
        return pd.DataFrame()
    
    percentile_ranges = divide_into_percentiles(merged_data['Win_Rate'])
    
    output_data = []
    for _, row in merged_data.iterrows():
        correlations = calculate_correlations(row['Win_Rate'], row['Arena Score'], 
                                              merged_data['Win_Rate'], merged_data['Arena Score'], 
                                              percentile_ranges)
        output_data.append({
            'Model': row['Model'],
            'Correlation_0-25%': correlations['0-25%'],
            'Correlation_25-50%': correlations['25-50%'],
            'Correlation_50-75%': correlations['50-75%'],
            'Correlation_75-100%': correlations['75-100%'],
            'Percentile_Range': percentile_ranges[percentile_ranges.index == row.name].iloc[0]
        })
    
    output = pd.DataFrame(output_data)
    return output

if __name__ == "__main__":
    alpaca_file = "existing-eval/alpaca-eval/data/alpaca_eval_scores.csv"
    arena_file = "chatbot-arena/data/rankings/chatbot_arena_scores.csv"
    result = main(alpaca_file, arena_file)
    print(result)
    result.to_csv("output_matrix.csv", index=False)