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

def get_kendall(x, y):
    x = x.to_numpy(dtype=float, copy=False)
    y = y.to_numpy(dtype=float, copy=False)
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    xtie = 0
    ytie = 0
    tot = 0.
    dis = 0.
    con = 0.
    
    for i in range(x.size):
        for j in range(i, x.size):
            if x[i] == x[j]:
                xtie += 1
            elif y[i] == y[j]:
                ytie += 1
            elif (x[i] > x[j]) and (y[i] > y[j]):
                con += 1
            elif (x[i] < x[j]) and (y[i] < y[j]):
                con += 1
            else:
                dis += 1
            
            tot += 1
    
    tau = (tot - xtie - ytie - 2 * dis) / np.sqrt((tot - xtie) * (tot - ytie))
    return tau

def calculate_quartile_correlations(model, original_leaderboard, model_leaderboard, quartiles):
    results = []
    model_counts = []
    for quartile in ['0-25%', '25-50%', '50-75%', '75-100%']:
        quartile_models = quartiles[quartiles == quartile].index
        common_models = list(set(quartile_models) & set(model_leaderboard.index))
        
        if len(common_models) > 1:
            original_ranks = original_leaderboard[common_models].rank(ascending=False, method='min')
            model_ranks = model_leaderboard[common_models].rank(ascending=False, method='min')
            
            tau = get_kendall(original_ranks, model_ranks)
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

def create_heatmap(df, title, filename, filter_low_counts=False):
    # Sort the dataframe by percentile in descending order
    df = df.sort_values('Percentile', ascending=False)
    
    plt.figure(figsize=(14, len(df) * 0.4))
    
    # Prepare data and annotations
    data = df.iloc[:, 1:5].values
    counts = df.iloc[:, -5:-1].values
    # print(counts)
    
    # Create annotations with both correlation and count
    annotations = []
    for i in range(data.shape[0]):
        row_annotations = []
        for j in range(data.shape[1]):
            if pd.isna(data[i, j]) or (filter_low_counts and counts[i, j] < 10):
                row_annotations.append('')
            else:
                row_annotations.append(f'{data[i, j]:.2f}\n({counts[i, j]})')
        annotations.append(row_annotations)
    
    # Create mask for cells with count < 10 if filtering
    mask = counts < 10 if filter_low_counts else np.zeros_like(counts, dtype=bool)
    
    # Create heatmap
    sns.heatmap(data, annot=annotations, fmt='', cmap='coolwarm', center=0, vmin=-1, vmax=1,
                yticklabels=[f"{row['Model']} ({row['Percentile']:.1f}%)" for _, row in df.iterrows()],
                xticklabels=['0-25%', '25-50%', '50-75%', '75-100%'], mask=mask)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Load data
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
results_df.to_csv('results/kt_new/model_quartile_correlations.csv', index=False)

# Calculate percentiles for each model
percentiles = original_leaderboard.rank(pct=True) * 100
results_df['Percentile'] = results_df['Model'].map(percentiles)

# Create heatmap with all data (no filtering)
create_heatmap(results_df, "Correlations by Quartile (All Data)", 'results/kt_new/correlations_heatmap_all.png', filter_low_counts=False)

# Create heatmap with filtered data (showing only cells with >= 10 battles)
create_heatmap(results_df, "Correlations by Quartile (>= 10 Battles)", 'results/kt_new/correlations_heatmap_filtered.png', filter_low_counts=True)

print("\nPlots saved as 'correlations_heatmap_all.png' and 'correlations_heatmap_filtered.png'")
print("Full results saved as 'model_quartile_correlations.csv'")