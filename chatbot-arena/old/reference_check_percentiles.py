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

# Compute overall leaderboard
overall_leaderboard = compute_mle_elo(battles)

def compute_model_specific_leaderboard(model_name):
    model_battles = battles[(battles['model_a'] == model_name) | (battles['model_b'] == model_name)]
    return compute_mle_elo(model_battles)

def compare_models_by_percentiles(model1, model2):
    model1_leaderboard = compute_model_specific_leaderboard(model1)
    model2_leaderboard = compute_model_specific_leaderboard(model2)
    
    common_models = list(set(overall_leaderboard.index) & set(model1_leaderboard.index) & set(model2_leaderboard.index))
    
    overall_ranks = overall_leaderboard[common_models].rank(ascending=False, method='min')
    model1_ranks = model1_leaderboard[common_models].rank(ascending=False, method='min')
    model2_ranks = model2_leaderboard[common_models].rank(ascending=False, method='min')
    
    percentiles = [25, 50, 75, 100]
    results = []
    
    for percentile in percentiles:
        threshold = np.percentile(overall_ranks, percentile)
        subset_models = overall_ranks[overall_ranks <= threshold].index
        
        tau1, _ = kendalltau(overall_ranks[subset_models], model1_ranks[subset_models])
        tau2, _ = kendalltau(overall_ranks[subset_models], model2_ranks[subset_models])
        
        results.append({
            'percentile': percentile,
            f'{model1}_correlation': tau1,
            f'{model2}_correlation': tau2
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['percentile'], results_df[f'{model1}_correlation'], label=model1)
    plt.plot(results_df['percentile'], results_df[f'{model2}_correlation'], label=model2)
    plt.xlabel('Percentile')
    plt.ylabel('Correlation with overall leaderboard')
    plt.title('Model Correlation by Percentile')
    plt.legend()
    plt.show()

compare_models_by_percentiles("gpt-4", "stablelm-tuned-alpha-7b")

def compare_all_models_by_percentiles():
    all_models = set(battles['model_a'].unique()) | set(battles['model_b'].unique())
    
    percentiles = [25, 50, 75, 100]
    results = {model: [] for model in all_models}
    
    for model in all_models:
        model_leaderboard = compute_model_specific_leaderboard(model)
        common_models = list(set(overall_leaderboard.index) & set(model_leaderboard.index))
        
        overall_ranks = overall_leaderboard[common_models].rank(ascending=False, method='min')
        model_ranks = model_leaderboard[common_models].rank(ascending=False, method='min')
        
        for percentile in percentiles:
            threshold = np.percentile(overall_ranks, percentile)
            subset_models = overall_ranks[overall_ranks <= threshold].index
            
            tau, _ = kendalltau(overall_ranks[subset_models], model_ranks[subset_models])
            
            results[model].append({
                'percentile': percentile,
                'correlation': tau
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {'model': model, 'percentile': item['percentile'], 'correlation': item['correlation']}
        for model, items in results.items()
        for item in items
    ])
    
    # Pivot the DataFrame for easier plotting
    results_pivot = results_df.pivot(index='percentile', columns='model', values='correlation')
    
    # Plotting
    plt.figure(figsize=(15, 10))
    for model in results_pivot.columns:
        plt.plot(results_pivot.index, results_pivot[model], label=model)
    
    plt.xlabel('Percentile')
    plt.ylabel('Correlation with overall leaderboard')
    plt.title('Model Correlation by Percentile for All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return results_pivot

# Call the new function
all_models_comparison = compare_all_models_by_percentiles()
print(all_models_comparison)