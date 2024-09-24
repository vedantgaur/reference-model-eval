import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
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

def compute_model_specific_leaderboard(model_name):
    model_battles = battles[(battles['model_a'] == model_name) | (battles['model_b'] == model_name)]
    return compute_mle_elo(model_battles)

def divide_into_quartiles(leaderboard):
    leaderboard_sorted = leaderboard.sort_values(ascending=False)
    quartiles = pd.qcut(leaderboard_sorted, q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    return pd.Series(quartiles, index=leaderboard_sorted.index)

def calculate_quartile_correlations(model, original_leaderboard, model_leaderboard, quartiles):
    results = []
    for quartile in ['0-25%', '25-50%', '50-75%', '75-100%']:
        quartile_models = quartiles[quartiles == quartile].index
        common_models = list(set(quartile_models) & set(model_leaderboard.index))
        
        if len(common_models) > 1:  # Need at least 2 models to calculate correlation
            original_ranks = original_leaderboard[common_models].rank(ascending=False, method='min')
            model_ranks = model_leaderboard[common_models].rank(ascending=False, method='min')
            
            tau, _ = kendalltau(original_ranks, model_ranks)
            results.append(tau)
        else:
            results.append(np.nan)
    
    if model in quartiles.index:
        results.append(quartiles[model])
    else:
        results.append('Not in leaderboard')
    
    return results

file_path = "/Users/vedantgaur/Projects/llm-eval/chatbot-arena/chatbot_arena.csv"
battles = pd.read_csv(file_path)

original_leaderboard = compute_mle_elo(battles)

all_models = pd.unique(battles[['model_a', 'model_b']].values.ravel())
print(f"Total number of models: {len(all_models)}")

quartiles = divide_into_quartiles(original_leaderboard)

results = []
for model in all_models:
    model_leaderboard = compute_model_specific_leaderboard(model)
    correlations = calculate_quartile_correlations(model, original_leaderboard, model_leaderboard, quartiles)
    results.append([model] + correlations)

columns = ['Model', '0-25%', '25-50%', '50-75%', '75-100%', 'Reference Model Quartile']
results_df = pd.DataFrame(results, columns=columns)

print("Quartile-based correlations:")
print(results_df)
results_df.to_csv('model_quartile_correlations.csv', index=False)

plt.figure(figsize=(12, len(results_df) * 0.3))
sns.heatmap(results_df.iloc[:, 1:-1], annot=True, cmap='coolwarm', center=0,
            yticklabels=results_df['Model'], xticklabels=columns[1:-1])
plt.title("Correlations by Quartile")
plt.tight_layout()
plt.show()