import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression
    
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    
    ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    if "tie" in df["winner"].unique():
        ptbl_tie = pd.pivot_table(
            df[df["winner"] == "tie"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            if np.isnan(ptbl_win.loc[m_a, m_b]) or np.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +np.log(BASE)
            X[cur_row, models[m_b]] = -np.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = np.log(BASE)
            X[cur_row + 1, models[m_b]] = -np.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

file_path = "/Users/vedantgaur/Projects/llm-eval/chatbot-arena/chatbot_arena.csv"
battles = pd.read_csv(file_path)

original_leaderboard = compute_mle_elo(battles)

def compute_model_specific_leaderboard(model_name):
    model_battles = battles[(battles['model_a'] == model_name) | (battles['model_b'] == model_name)]
    return compute_mle_elo(model_battles)

all_models = pd.unique(battles[['model_a', 'model_b']].values.ravel())

model_counts = battles['model_a'].value_counts() + battles['model_b'].value_counts()
top_20_models = model_counts.nlargest(20).index.tolist()

correlations = {}
for model in top_20_models:
    model_leaderboard = compute_model_specific_leaderboard(model)
    
    common_models = list(set(original_leaderboard.index) & set(model_leaderboard.index))

    original_ranks = original_leaderboard[common_models].rank(ascending=False, method='min')
    model_ranks = model_leaderboard[common_models].rank(ascending=False, method='min')
    
    tau, _ = kendalltau(original_ranks, model_ranks)
    correlations[model] = tau

results_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
results_df = results_df.sort_values('Correlation', ascending=False)

print("Kendall's Tau correlations:")
print(results_df)

plt.figure(figsize=(15, 8))
plt.bar(results_df.index, results_df['Correlation'])
plt.title("Kendall's Tau Correlation with Original Leaderboard")
plt.xlabel("Model")
plt.ylabel("Correlation")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

results_df.to_csv('model_correlations.csv')