import pandas as pd

def analyze_correlations(results_df):
    highest_correlations = {}
    overall_highest = {'model': '', 'value': 0}
    
    for quartile in results_df.columns[1:-1]:  
        max_corr_value = results_df[quartile].max()
        models_with_max_corr = results_df[results_df[quartile] == max_corr_value]['Model'].tolist()
        highest_correlations[quartile] = (models_with_max_corr, max_corr_value)
    
    results_df['Average Correlation'] = results_df.iloc[:, 1:-1].mean(axis=1)
    overall_highest['model'] = results_df.loc[results_df['Average Correlation'].idxmax(), 'Model']
    overall_highest['value'] = results_df['Average Correlation'].max()
    
    return highest_correlations, overall_highest


results_df = pd.read_csv("model_quartile_correlations.csv")
highest_corrs, overall_highest = analyze_correlations(results_df)


print("Highest correlations with each quartile:")
for quartile, (models, value) in highest_corrs.items():
    print(f"{quartile}: {models} (correlation: {value:.2f})")

print(f"\nOverall highest correlation model: {overall_highest['model']} (average correlation: {overall_highest['value']:.2f})")