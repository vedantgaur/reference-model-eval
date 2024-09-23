import pandas as pd
import numpy as np
from bert_score import score
from tqdm import tqdm

def compute_bertscore(row):
    conv_a = ' '.join([turn['content'] for turn in eval(row['conversation_a'])])
    conv_b = ' '.join([turn['content'] for turn in eval(row['conversation_b'])])
    P, R, F1 = score([conv_a], [conv_b], lang="en", verbose=False)
    return F1.item()

df_arena = pd.read_csv('~/projects/llm-eval/chatbot-arena/filtered_chatbot_arena.csv')
df_gpt4 = pd.read_csv('~/projects/llm-eval/chatbot-arena/runs/filtered_results.csv')

tqdm.pandas(desc="Computing BertScores")
df_arena['bertscore'] = df_arena.progress_apply(compute_bertscore, axis=1)

df_sorted = df_arena.sort_values('bertscore')
bottom_25 = df_sorted.iloc[:len(df_sorted)//4]
top_25 = df_sorted.iloc[-len(df_sorted)//4:]

def check_disagreement(arena_row, gpt4_row):
    human_judgment = arena_row['winner']
    gpt4_judgment = gpt4_row['final response']
    return human_judgment != gpt4_judgment

bottom_25_disagreement = sum(check_disagreement(arena_row, df_gpt4.loc[arena_row.name]) 
                             for _, arena_row in bottom_25.iterrows()) / len(bottom_25)
top_25_disagreement = sum(check_disagreement(arena_row, df_gpt4.loc[arena_row.name]) 
                          for _, arena_row in top_25.iterrows()) / len(top_25)

print(f"Disagreement in bottom 25%: {bottom_25_disagreement:.2%}")
print(f"Disagreement in top 25%: {top_25_disagreement:.2%}")

from scipy import stats

bottom_disagreements = [check_disagreement(arena_row, df_gpt4.loc[arena_row.name]) 
                        for _, arena_row in bottom_25.iterrows()]
top_disagreements = [check_disagreement(arena_row, df_gpt4.loc[arena_row.name]) 
                     for _, arena_row in top_25.iterrows()]

t_stat, p_value = stats.ttest_ind(bottom_disagreements, top_disagreements)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")