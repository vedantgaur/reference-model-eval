import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    all_models = pd.unique(df[['model_a', 'model_b']].values.ravel())
    all_models = [model for model in all_models if isinstance(model, str)]
    
    ptbl_win = pd.DataFrame(0, index=all_models, columns=all_models)
    
    for _, row in df.iterrows():
        if row['winner_model_a'] == 1:
            ptbl_win.loc[row['model_a'], row['model_b']] += 2
        elif row['winner_model_b'] == 1:
            ptbl_win.loc[row['model_b'], row['model_a']] += 2
        elif row['winner_tie'] == 1:
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

def compute_model_specific_leaderboard(model_name):
    model_battles = battles[(battles['model_a'] == model_name) | (battles['model_b'] == model_name)]
    return compute_mle_elo(model_battles)

def divide_into_quartiles(leaderboard):
    leaderboard_sorted = leaderboard.sort_values(ascending=False)
    quartiles = pd.qcut(leaderboard_sorted, q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    return pd.Series(quartiles, index=leaderboard_sorted.index)

def calculate_quartile_correlations(model, original_leaderboard, model_leaderboard, quartiles):
    results = []
    model_counts = []
    for quartile in ['0-25%', '25-50%', '50-75%', '75-100%']:
        quartile_models = quartiles[quartiles == quartile].index
        common_models = list(set(quartile_models) & set(model_leaderboard.index))
        
        if len(common_models) > 1:
            original_ranks = original_leaderboard[common_models].rank(ascending=False, method='min')
            model_ranks = model_leaderboard[common_models].rank(ascending=False, method='min')
            
            tau, _ = kendalltau(original_ranks, model_ranks)
            results.append(tau)
            model_counts.append(len(common_models))
        else:
            results.append(np.nan)
            model_counts.append(0)
    
    if model in quartiles.index:
        results.append(quartiles[model])
    else:
        results.append('Not in leaderboard')
    
    return results, model_counts

def analyze_correlations(results_df):
    highest_correlations = {}
    overall_highest = {'model': '', 'value': 0}
    
    for quartile in results_df.columns[1:-1:2]:
        max_corr_value = results_df[quartile].max()
        models_with_max_corr = results_df[results_df[quartile] == max_corr_value]['Model'].tolist()
        model_count = results_df[f'{quartile}_count'].iloc[0]
        highest_correlations[quartile] = (models_with_max_corr, max_corr_value, model_count)
    
    results_df['Average Correlation'] = results_df.iloc[:, 1:-1:2].mean(axis=1)
    overall_highest['model'] = results_df.loc[results_df['Average Correlation'].idxmax(), 'Model']
    overall_highest['value'] = results_df['Average Correlation'].max()
    
    return highest_correlations, overall_highest

hf_data = load_dataset("lmsys/lmsys-arena-human-preference-55k")
battles = pd.concat([hf_data[split].to_pandas() for split in hf_data.keys()])

print("First few rows of the dataset:")
print(battles.head())

original_leaderboard = compute_mle_elo(battles)

all_models = pd.unique(battles[['model_a', 'model_b']].values.ravel())
all_models = np.array([model for model in all_models if isinstance(model, str)])
print(f"\nTotal number of models: {len(all_models)}")
print("All models:")
print(all_models)

quartiles = divide_into_quartiles(original_leaderboard)

results = []
for model in all_models:
    model_leaderboard = compute_model_specific_leaderboard(model)
    correlations, model_counts = calculate_quartile_correlations(model, original_leaderboard, model_leaderboard, quartiles)
    results.append([model] + correlations + model_counts)

columns = ['Model', '0-25%', '25-50%', '50-75%', '75-100%', 'Reference Model Quartile', 
           '0-25%_count', '25-50%_count', '50-75%_count', '75-100%_count']
results_df = pd.DataFrame(results, columns=columns)

print("\nQuartile-based correlations:")
print(results_df.to_string())
results_df.to_csv('model_quartile_correlations.csv', index=False)

highest_corrs, overall_highest = analyze_correlations(results_df)

print("\nHighest correlations with each quartile:")
for quartile, (models, value, count) in highest_corrs.items():
    print(f"{quartile}: {models} (correlation: {value:.2f}, models compared: {count})")

print(f"\nOverall highest correlation model: {overall_highest['model']} (average correlation: {overall_highest['value']:.2f})")

plt.figure(figsize=(12, len(results_df) * 0.3))
sns.heatmap(results_df.iloc[:, 1:-5], annot=True, cmap='coolwarm', center=0,
            yticklabels=results_df['Model'], xticklabels=columns[1:-5])
plt.title("Correlations by Quartile")
plt.tight_layout()
plt.savefig('correlations_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPlot saved as 'correlations_heatmap.png'")
print("Full results saved as 'model_quartile_correlations.csv'")