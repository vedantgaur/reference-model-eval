import pandas as pd
import numpy as np
from scipy.stats._stats import _kendall_dis
from scipy.stats import _mstats_basic as mstats_basic
from scipy._lib._bunch import _make_tuple_bunch


"""SignificanceResult = _make_tuple_bunch('SignificanceResult',
                                       ['statistic', 'pvalue'], [])"""


def get_kendall(x, y, apply_dd, apply_dd_val, variant='b'):
    x = x.to_numpy(dtype=float, copy=False)
    y = y.to_numpy(dtype=float, copy=False)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    xtie=0
    ytie=0
    ntie=0.
    tot=0.
    dis=0.
    con=0.

    for i in range(x.size):
        for j in range(i, x.size):
            if apply_dd:
                if abs(y[i] - y[j]) > apply_dd_val:
                    continue

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
            

    #print(xtie, ytie, ntie, tot, dis, con)
    #print(tot)
    tau = (tot - xtie - ytie - 2 * dis)/np.sqrt((tot - xtie)*(tot - ytie))
    #print(tau)
    return tau#, tot

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        # Python ints to avoid overflow down the line
        return (int((cnt * (cnt - 1) // 2).sum()),
                int((cnt * (cnt - 1.) * (cnt - 2)).sum()),
                int((cnt * (cnt - 1.) * (2*cnt + 5)).sum()))


    size = x.size
    print(y)
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)


    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)



    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        res= np.nan
        return res

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    if variant == 'b':
        tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == 'c':
        minclasses = min(len(set(x)), len(set(y)))
        tau = 2*con_minus_dis / (size**2 * (minclasses-1)/minclasses)
    else:
        raise ValueError(f"Unknown variant of the method chosen: {variant}. "
                         "variant must be 'b' or 'c'.")

    # Limit range to fix computational errors
    tau = np.minimum(1., max(-1., tau))

    return tau

def custom_corr(df, n_min_models=25, apply_dd=False, apply_dd_val=0.5):
    columns = df.columns
    columns = [col for col in columns if col not in ['Arena Elo']]
    
    filtered_cols = []
    for c in columns:
        temp_df = df[['Arena Elo', c]].dropna()
        if len(temp_df) >= n_min_models -1:
            filtered_cols.append(c)
    
    mat = pd.DataFrame(np.zeros(len(filtered_cols)), index=filtered_cols, columns=[apply_dd_val])
    for c in filtered_cols:
        df_filtered = df[['Arena Elo', c]].dropna()
        k = get_kendall(df_filtered['Arena Elo'], df_filtered[c], apply_dd, apply_dd_val)
        mat.loc[c, apply_dd_val] = k
    return mat
        

if __name__=='__main__':
    is_drop_alpaca = True
    n_min_models = 25


    min_periods = n_min_models -1
    filename="~/projects/llm-eval/existing-eval/new_new_benchmarks.csv"
    df = pd.read_csv(filename, index_col=0)
    df.columns = [c.split("\n")[0] for c in df.columns]

    df = df.iloc[:, [i for i in range(df.columns.size) if i != 1]]
    #print(df)

    """if is_drop_alpaca:
        df = df.drop(columns=["AlpacaEval 2.0", "AlpacaEval 1.0"])"""
    
    df_corr = df.corr(method="kendall", min_periods=min_periods).dropna(how="all", axis=0).dropna(how="all", axis=1)['Arena Elo']
    #print(df_corr)

    mat_sys = custom_corr(df, n_min_models, apply_dd_val='Original')
    dd_vals = [0.3, 0.5, 1, 2, 5, 10]
    for dval in dd_vals:
        mat_sys_custom = custom_corr(df, n_min_models, apply_dd=True, apply_dd_val=dval)
        mat_sys = pd.concat([mat_sys, mat_sys_custom], axis=1, join='inner')
    
    print(mat_sys)
