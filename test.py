import pandas as pd

df = pd.read_csv("./chatbot-arena/data/chatbot_arena.csv")

full_models = list(df["model_a"]) + list(df["model_b"])

unique_models = set(full_models)

print(unique_models)
print(len(unique_models))
print(len(df))