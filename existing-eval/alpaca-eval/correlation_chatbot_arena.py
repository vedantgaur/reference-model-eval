import numpy as np
from scipy.stats import kendalltau

model_comparison = {
    "gpt-4o-2024-05-13": [0.50, 0.19, 0.06, 0.09, 0.25, 0.21, 0.26, 0.36],
    "gpt-3.5-turbo-1106": [0.75, 0.50, 0.13, 0.22, 0.45, 0.34, 0.54, 0.67],
    "llama-2-7b-chat-hf": [0.93, 0.88, 0.50, 0.60, 0.79, 0.78, 0.84, 0.88],
    "llama-2-13b-chat-hf": [0.90, 0.81, 0.39, 0.50, 0.74, 0.70, 0.79, 0.84],
    "Meta-Llama-3-8B-Instruct": [0.74, 0.58, 0.22, 0.28, 0.50, 0.44, 0.58, 0.64],
    "Mistral-7B-Instruct-v0.2": [0.81, 0.49, 0.24, 0.30, 0.56, 0.50, 0.60, 0.68],
    "Mistral-8x7B-Instruct-v0.1": [0.74, 0.46, 0.17, 0.24, 0.47, 0.44, 0.50, 0.59],
    "Mixtral-8x22B-Instruct-v0.1": [0.66, 0.36, 0.14, 0.16, 0.37, 0.34, 0.44, 0.50]
}

chatbot_arena_ratings = {
    "gpt-4o-2024-05-13": 1287,
    "gpt-3.5-turbo-1106": 1068,
    "llama-2-7b-chat-hf": 1037,
    "llama-2-13b-chat-hf": 1063,
    "Meta-Llama-3-8B-Instruct": 1152,
    "Mistral-7B-Instruct-v0.2": 1072,
    "Mistral-8x7B-Instruct-v0.1": 1114,
    "Mixtral-8x22B-Instruct-v0.1": 1146
}

models = list(model_comparison.keys())
values = list(model_comparison.values())

matrix = np.array(values)

elo_ratings = [chatbot_arena_ratings[model] for model in models if model in chatbot_arena_ratings]

tau_results = []
for i in range(len(models)):
    if models[i] in chatbot_arena_ratings:
        tau, _ = kendalltau(matrix[i], elo_ratings)
        tau_results.append((models[i], tau))

print("Kendall's tau correlation results with Chatbot Arena Elo ratings:")
for model, tau in tau_results:
    print(f"{model}: {tau:.4f}")