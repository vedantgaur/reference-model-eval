import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    all_models = pd.unique(df[['model_a', 'model_b']].values.ravel())
    all_models = [model for model in all_models if isinstance(model, str)]
    
    ptbl_win = pd.DataFrame(0, index=all_models, columns=all_models)
    
    for _, row in df.iterrows():
        if row['winner'] == 'model_a':
            ptbl_win.loc[row['model_a'], row['model_b']] += 2
        elif row['winner'] == 'model_b':
            ptbl_win.loc[row['model_b'], row['model_a']] += 2
        elif row['winner'] == 'tie':
            ptbl_win.loc[row['model_a'], row['model_b']] += 1
            ptbl_win.loc[row['model_b'], row['model_a']] += 1

    models = pd.Series(np.arange(len(all_models)), index=all_models)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in all_models:
        for m_b in all_models:
            if m_a == m_b:
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

battles = pd.read_csv("./chatbot-arena/data/chatbot_arena.csv")
original_leaderboard = compute_mle_elo(battles)
print(original_leaderboard)

all_models = pd.unique(battles[['model_a', 'model_b']].values.ravel())
all_models = [model for model in all_models if isinstance(model, str)]

def compute_model_specific_leaderboard(model_name):
    model_battles = battles[(battles['model_a'] == model_name) | (battles['model_b'] == model_name)]
    return compute_mle_elo(model_battles)

correlations = {}
for model in all_models:
    model_leaderboard = compute_model_specific_leaderboard(model)
    common_models = list(set(original_leaderboard.index) & set(model_leaderboard.index))
    correlation = pearsonr(original_leaderboard[common_models], model_leaderboard[common_models])[0]
    correlations[model] = correlation

correlation_series = pd.Series(correlations)

percentile_ranges = pd.qcut(original_leaderboard, q=4, labels=['0-25', '25-50', '50-75', '75-100'])

result_matrix = pd.DataFrame(index=all_models, columns=['0-25', '25-50', '50-75', '75-100', 'model_percentile'])

for percentile_range in ['0-25', '25-50', '50-75', '75-100']:
    subset_models = original_leaderboard[percentile_ranges == percentile_range].index
    subset_correlations = correlation_series[subset_models]
    result_matrix.loc[subset_models, percentile_range] = subset_correlations

result_matrix['model_percentile'] = percentile_ranges

print(result_matrix)
result_matrix.to_csv('result_matrix.csv')