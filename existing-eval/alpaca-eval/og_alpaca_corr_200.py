import numpy as np
from scipy.stats import pearsonr, spearmanr

# Model scores
test_models_arena = {
    'llama-2-13b-chat-hf': 1063,
    'zephyr-7b-beta': 1054,
    'qwen1.5-7b-chat': 1070,
    'guanaco-33b': 1033,
    'vicuna-13b': 1042,
    'zephyr-7b-alpha': 1041,
    'qwen-14b-chat': 1035,
    'mistral-7b-instruct-v0.3': 1072,
    'vicuna-7b': 1005,
    'chatglm2-6b': 924
}

# Update the test_models_alpaca with new win rates
test_models_alpaca = {
    'chatglm2-6b': 2.0,  # Non-length controlled
    'guanaco-33b': 7.5,  # Non-length controlled
    'llama-2-13b-chat-hf': 8.5,  # Non-length controlled
    'mistral-7b-instruct-v0.3': 22.25,  # Non-length controlled
    'qwen-14b-chat': 5.75,  # Non-length controlled
    'qwen1.5-7b-chat': 19.25,  # Non-length controlled
    'vicuna-7b': 1.5,  # Non-length controlled
    'vicuna-13b': 4.75,  # Non-length controlled
    'zephyr-7b-alpha': 11.0,  # Non-length controlled
    'zephyr-7b-beta': 14.0  # Non-length controlled
}

# New length controlled scores
length_controlled_scores = {
    'chatglm2-6b': 5.6374361669805815,
    'guanaco-33b': 10.14761803014255,
    'llama-2-13b-chat-hf': 9.736368412724719,
    'mistral-7b-instruct-v0.3': 32.78508622486959,
    'qwen-14b-chat': 10.692519833473895,
    'qwen1.5-7b-chat': 20.2390205736907,
    'vicuna-7b': 3.4157234491065025,
    'vicuna-13b': 10.360113065879933,
    'zephyr-7b-alpha': 20.180754426245095,
    'zephyr-7b-beta': 20.007725341006807
}

# Ensure the order of models is the same for both datasets
models = list(test_models_arena.keys())
arena_scores = [test_models_arena[model] for model in models]
alpaca_scores = [test_models_alpaca[model] for model in models]
length_controlled_alpaca_scores = [length_controlled_scores[model] for model in models]

# Calculate Pearson correlation for non-length controlled
pearson_corr, pearson_p = pearsonr(arena_scores, alpaca_scores)

# Calculate Spearman correlation for non-length controlled
spearman_corr, spearman_p = spearmanr(arena_scores, alpaca_scores)

# Calculate Pearson correlation for length controlled
length_controlled_pearson_corr, length_controlled_pearson_p = pearsonr(arena_scores, length_controlled_alpaca_scores)

# Calculate Spearman correlation for length controlled
length_controlled_spearman_corr, length_controlled_spearman_p = spearmanr(arena_scores, length_controlled_alpaca_scores)

print("Correlation between ChatBot Arena and Alpaca scores (non-length controlled):")
print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

print("\nCorrelation between ChatBot Arena and Alpaca scores (length controlled):")
print(f"Pearson correlation: {length_controlled_pearson_corr:.4f} (p-value: {length_controlled_pearson_p:.4f})")
print(f"Spearman correlation: {length_controlled_spearman_corr:.4f} (p-value: {length_controlled_spearman_p:.4f})")

# Print the data for verification
print("\nModel Scores:")
print("Model                    Arena   Alpaca (Non-length controlled)   Alpaca (Length controlled)")
print("---------------------------------------------------------------------------------------------")
for model in models:
    print(f"{model:<22} {test_models_arena[model]:<7} {test_models_alpaca[model]:<7} {length_controlled_scores[model]:<7}")
