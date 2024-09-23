import pandas as pd
import json

pd.read_csv("/Users/vedantgaur/Projects/llm-eval/chatbot-arena/chatbot_arena.csv")

sampled_data = pd.read_csv("/Users/vedantgaur/Projects/llm-eval/chatbot-arena/chatbot_arena.csv")
sampled_data = sampled_data[['winner', 'conversation_a', 'conversation_b', 'model_a', 'model_b']]

model_a = "gpt-4"
model_b = "claude-v1"

sampled_data = sampled_data[(sampled_data['model_a'] == model_a) & (sampled_data['model_b'] == model_b) | (sampled_data['model_a'] == model_b) & (sampled_data['model_b'] == model_a)]
sampled_data = sampled_data[(sampled_data['winner'] != 'tie') & (sampled_data['winner'] != 'tie (bothbad)')]
sampled_data = sampled_data.sample(n=len(sampled_data), random_state=42)
print(sampled_data)

wins = {
    model_a: 0,
    model_b: 0
}

gpt4_wins = 0
for index, row in sampled_data.iterrows():
    if row['winner'] == 'model_a' and row['model_a'] == 'gpt-4':
        gpt4_wins += 1
        if gpt4_wins <= 23:
            sampled_data.drop(index, inplace=True)
print(len(sampled_data))

for index, row in sampled_data.iterrows():
    winner = row['winner']
    if winner == "model_a":
        wins[row["model_a"]] += 1
    elif winner == "model_b":
        wins[row["model_b"]] += 1

print(wins)

num_case_1, num_case_2 = 0, 0
tot_case_1, tot_case_2 = 0, 0

x_num_case_1, x_num_case_2 = 0, 0
x_tot_case_1, x_tot_case_2 = 0, 0

print(sampled_data)
for index, row in sampled_data.iterrows():
    winner = row['winner']
    if winner == "model_a":
        model_value = row['model_a']
    elif winner == "model_b":
        model_value = row['model_b']

    conv_a, conv_b = eval(row["conversation_a"]), eval(row["conversation_b"])
    conv_a_len, conv_b_len = 0, 0

    for part in conv_a:
        if part["role"] == "assistant":
            data = part["content"]
            conv_a_len += len(data)
    for part in conv_b:
        if part["role"] == "assistant":
            data = part["content"]
            conv_b_len += len(data)
    
    if winner == "model_a" and conv_a_len > conv_b_len:
        tot_case_1 += 1
        if model_value == "gpt-4":
            num_case_1 += 1
    elif winner == "model_b" and conv_b_len > conv_a_len:
        tot_case_1 += 1
        if model_value == "gpt-4":
            num_case_1 += 1
    elif winner == "model_a" and conv_a_len <= conv_b_len:
        tot_case_2 += 1
        if model_value == "gpt-4":
            num_case_2 += 1
    elif winner == "model_b" and conv_b_len <= conv_a_len:
        tot_case_2 += 1
        if model_value == "gpt-4":
            num_case_2 += 1

print(num_case_1, num_case_2, tot_case_1, tot_case_2)

print(num_case_1/tot_case_1-num_case_2/tot_case_2)