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

test_models_alpaca = {
    'llama-2-13b-chat-hf': 7.7,
    'zephyr-7b-beta': 11.0,
    'qwen1.5-7b-chat': 0.147,
    'guanaco-33b': 5.0,
    'vicuna-13b': 5.8,
    'zephyr-7b-alpha': 8.4,
    'qwen-14b-chat': 7.5,
    'mistral-7b-instruct-v0.3': 16.7,
    'vicuna-7b': 4.2,
    'chatglm2-6b': 2.8
}

# Ensure the order of models is the same for both datasets
models = list(test_models_arena.keys())
arena_scores = [test_models_arena[model] for model in models]
alpaca_scores = [test_models_alpaca[model] for model in models]

# Calculate Pearson correlation
pearson_corr, pearson_p = pearsonr(arena_scores, alpaca_scores)

# Calculate Spearman correlation
spearman_corr, spearman_p = spearmanr(arena_scores, alpaca_scores)

print("Correlation between ChatBot Arena and Alpaca scores:")
print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

# Print the data for verification
print("\nModel Scores:")
print("Model                    Arena   Alpaca")
print("----------------------------------------")
for model in models:
    print(f"{model:<22} {test_models_arena[model]:<7} {test_models_alpaca[model]:<7}")
# from scipy.stats import pearsonr, spearmanr

# test_models_arena = {
#     'llama-2-13b-chat-hf': 1063,
#     'zephyr-7b-beta': 1054,
#     'qwen1.5-7b-chat': 1070,
#     'guanaco-33b': 1033,
#     'vicuna-13b': 1042,
#     'zephyr-7b-alpha': 1041,
#     'qwen-14b-chat': 1035,
#     'mistral-7b-instruct-v0.3': 1072,
#     'vicuna-7b': 1005,
#     'chatglm2-6b': 924
# }

# test_models_alpaca = {
#     'llama-2-13b-chat-hf': 8.4,
#     'zephyr-7b-beta': 13.2,
#     'qwen1.5-7b-chat': 14.7,
#     'guanaco-33b': 5.7,
#     'vicuna-13b': 9.2,
#     'zephyr-7b-alpha': 10.3,
#     'qwen-14b-chat': 12.4,
#     'mistral-7b-instruct-v0.3': 20.6,
#     'vicuna-7b': 6.3,
#     'chatglm2-6b': 4.4
# }

# # Convert dictionaries to lists for correlation calculation
# arena_values = list(test_models_arena.values())
# alpaca_values = list(test_models_alpaca.values())

# # Compute Pearson correlation
# pearson_corr, _ = pearsonr(arena_values, alpaca_values)
# print(f'Pearson correlation: {pearson_corr}')

# # Compute Spearman correlation
# spearman_corr, _ = spearmanr(arena_values, alpaca_values)
# print(f'Spearman correlation: {spearman_corr}')
