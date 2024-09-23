import pandas as pd

gpt_res = pd.read_csv("/Users/vedantgaur/Projects/llm-eval/chatbot-arena/runs/results2.csv")
gpt_res = gpt_res[['final response', 'flipped']]

# print(gpt_res)

sampled_data = pd.read_csv("/Users/vedantgaur/Projects/llm-eval/chatbot-arena/chatbot_arena.csv")
sampled_data = sampled_data[['conversation_a', 'conversation_b', 'model_a', 'model_b']]

merged_data = pd.merge(gpt_res, sampled_data, left_index=True, right_index=True)

# print(merged_data[['final response', 'flipped']])

def flip_result(row):
    # print(row['final response'])
    if row['flipped']:
        if row['final response'] == 'model_a':
            return 'model_b'
        elif row['final response'] == 'model_b':
            return 'model_a'
    return row['final response']

merged_data['winner'] = merged_data.apply(flip_result, axis=1)

merged_data = merged_data[(merged_data['winner'] != 'tie') & (merged_data['winner'] != 'tie (bothbad)')]
print(len(merged_data.dropna()))

model_a = "gpt-4"
model_b = "claude-v1"

filtered_data = merged_data[
    ((merged_data['model_a'] == model_a) & (merged_data['model_b'] == model_b)) |
    ((merged_data['model_a'] == model_b) & (merged_data['model_b'] == model_a))
]
print(filtered_data)

wins = {model_a: 0, model_b: 0}

for index, row in filtered_data.iterrows():
    winner = row['winner']
    if winner == "model_a":
        wins[row["model_a"]] += 1
    elif winner == "model_b":
        wins[row["model_b"]] += 1

print("Win counts:", wins)

num_case_1, num_case_2 = 0, 0
tot_case_1, tot_case_2 = 0, 0

for index, row in filtered_data.iterrows():
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

print("Analysis results:")
print(f"Case 1: {num_case_1}/{tot_case_1}")
print(f"Case 2: {num_case_2}/{tot_case_2}")

if tot_case_1 > 0 and tot_case_2 > 0:
    print(f"Difference: {num_case_1/tot_case_1 - num_case_2/tot_case_2}")
else:
    print("Not enough data for analysis")