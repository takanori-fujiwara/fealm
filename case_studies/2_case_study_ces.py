import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

from umap import UMAP
import fealm.plot as fplot

import json
from output_formatter import dump_all

q2type = {
    'CC20.302': 'ordinal',
    'CC20.307': 'ordinal',
    'CC20.320a': 'ordinal',
    'CC20.327a': 'binary',
    'CC20.327d': 'binary',
    'CC20.327e': 'binary',
    'CC20.330b': 'binary',
    'CC20.330c': 'binary',
    'CC20.331a': 'binary',
    'CC20.331b': 'binary',
    'CC20.331c': 'binary',
    'CC20.331d': 'binary',
    'CC20.331e': 'binary',
    'CC20.332a': 'binary',
    'CC20.332b': 'binary',
    'CC20.332c': 'binary',
    'CC20.332d': 'binary',
    'CC20.332e': 'binary',
    'CC20.332f': 'binary',
    'CC20.333a': 'binary',
    'CC20.333b': 'binary',
    'CC20.333c': 'binary',
    'CC20.333d': 'binary',
    'CC20.334a': 'binary',
    'CC20.334b': 'binary',
    'CC20.334c': 'binary',
    'CC20.334d': 'binary',
    'CC20.334e': 'binary',
    'CC20.334f': 'binary',
    'CC20.334g': 'binary',
    'CC20.334h': 'binary',
    'CC20.338a': 'binary',
    'CC20.338b': 'binary',
    'CC20.338c': 'binary',
    'CC20.338d': 'binary',
    'CC20.340a': 'ordinal',
    'CC20.350a': 'binary',
    'CC20.350b': 'binary',
    'CC20.350c': 'binary',
    'CC20.350d': 'binary',
    'CC20.350e': 'binary',
    'CC20.350f': 'binary',
    'CC20.350g': 'binary',
    'CC20.355a': 'binary',
    'CC20.355b': 'binary',
    'CC20.355c': 'binary',
    'CC20.355d': 'binary',
    'CC20.355e': 'binary',
    'CC20.356': 'binary',
    'ideo5': 'ordinal',
    'CC20.440a': 'ordinal',
    'CC20.440b': 'ordinal',
    'CC20.440c': 'ordinal',
    'CC20.440d': 'ordinal',
    'CC20.441a': 'ordinal',
    'CC20.441b': 'ordinal',
    'CC20.441e': 'ordinal',
    'CC20.441f': 'ordinal',
    'CC20.441g': 'ordinal',
    'CC20.442a': 'binary',
    'CC20.442b': 'binary',
    'CC20.442c': 'binary',
    'CC20.442d': 'binary',
    'CC20.442e': 'binary',
    'CC20.443.1': 'ordinal',
    'CC20.443.2': 'ordinal',
    'CC20.443.3': 'ordinal',
    'CC20.443.4': 'ordinal',
    'CC20.443.5': 'ordinal',
    'label': 'categorical'
}

ordinal_cols = []
for key in q2type:
    if q2type[key] == 'ordinal':
        ordinal_cols.append(key)
ordinal_cols.append('label')


def process_cesdata(df,
                    method='dropna',
                    na_categories={
                        'CC20_302': 6,
                        'CC20_320a': 5,
                        'CC20_340a': 8,
                        'CC20_356': 3,
                        'ideo5': 6
                    }):
    ### method={'fillna', 'fillna_with_mode', 'dropna'}
    for col in na_categories:
        df.loc[df[col] == na_categories[col], col] = np.nan

    if method == 'fillna':
        fillna_based_on_dtype(df)
    elif method == 'fillna_with_mode':
        df = df.fillna(df.mode().iloc[0])
    elif method == 'dropna':
        df = df.dropna()

    df = df.astype(int)

    df.columns = df.columns.str.replace('_', '.')
    X_dem = df[df['CC20.433a'] == 1].drop(columns=['CC20.433a'])
    X_rep = df[df['CC20.433a'] == 2].drop(columns=['CC20.433a'])
    X_ind = df[df['CC20.433a'] == 3].drop(columns=['CC20.433a'])
    X_oth = df[df['CC20.433a'] == 4].drop(columns=['CC20.433a'])

    X = pd.concat([X_dem, X_rep, X_ind, X_oth])
    X['label'] = ['Dem'] * X_dem.shape[0] + ['Rep'] * X_rep.shape[0] + [
        'Ind'
    ] * X_ind.shape[0] + ['Oth'] * X_oth.shape[0]

    return X


if __name__ == '__main__':
    df = pd.read_csv('./data/ces2020.csv')
    df = process_cesdata(df, method='dropna')

    df = df[ordinal_cols]

    y = df.iloc[:, -1]
    for class_id, target_name in zip([0, 1, 2, 3],
                                     ['Dem', 'Rep', 'Ind', 'Oth']):
        y = np.where(y == target_name, class_id, y)
    y = y.astype('int')

    df.drop(['label'], axis=1, inplace=True)

    # filter out highly correlated attrs (to perform better optimization)
    # plt.matshow(df.corr() >= 0.7)
    # plt.colorbar()
    # plt.show()
    upper_tri = df.corr().abs().where(
        np.triu(np.ones(df.corr().abs().shape), k=1).astype(bool))
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] >= 0.7)
    ]
    df.drop(to_drop, axis=1, inplace=True)

    X = np.array(df)
    inst_names = df.index
    feat_names = np.array(df.columns)
    target_names = np.array(['Dem', 'Rep'])
    target_colors = np.array(['#507AA6', '#DF585C'])

    # select only Dem and Rep
    np.random.seed(10)
    idx_y0 = np.where(y == 0)[0]
    idx_y1 = np.where(y == 1)[0]

    # make even # of samples
    idx_y0 = np.random.randint(len(idx_y0), size=len(idx_y1))
    idx = np.concatenate((idx_y0, idx_y1), axis=0)

    X = X[idx, :]
    inst_names = inst_names[idx]
    y = y[idx]

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    n_neighbors = 20  # 50
    min_dist = 0.1

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X)
    fplot.plot_embeddings([Y], y)
    plt.show()

    from fealm.fealm import FEALM
    from fealm.optimizer import AdaptiveNelderMead

    forms_to_settings = {
        'w': {
            'n_repeats': 10,
            'n_components': None,
            'max_cost_evaluations': 2000,
            'lasso_coeff': -300,
            'ridge_coeff': 0
        },
        'p_wMv': {
            'n_repeats': 5,
            'n_components': 3,
            'max_cost_evaluations': 4000,
            'lasso_coeff': 100,
            'ridge_coeff': 0
        }
    }
    n_jobs = -1
    max_time = 10800

    Ps = []
    for form in forms_to_settings:
        optimizer = AdaptiveNelderMead(
            max_cost_evaluations=forms_to_settings[form]
            ['max_cost_evaluations'],
            max_time=max_time,
            n_jobs=n_jobs)

        fealm = FEALM(n_neighbors=n_neighbors,
                      projection_form=form,
                      n_components=forms_to_settings[form]['n_components'],
                      n_repeats=forms_to_settings[form]['n_repeats'],
                      optimizer=optimizer,
                      lasso_coeff=forms_to_settings[form]['lasso_coeff'],
                      ridge_coeff=forms_to_settings[form]['ridge_coeff'])
        fealm = fealm.fit(X)
        Ps += fealm.Ps

    P0 = np.diag([1] * X.shape[1])
    cluster_result = fealm.find_representative_Ps(
        X,
        XP_dr_inst=umap,
        Ps=Ps + [P0],
        n_representatives=10 if len(Ps) > 10 else len(Ps))

    repr_indices = cluster_result['repr_indices']
    Ys = cluster_result['Ys']
    emb_of_Ys = cluster_result['emb_of_Ys']
    cluster_ids = cluster_result['cluster_ids']
    repr_Ys = cluster_result['repr_Ys']

    result = dump_all(X, y, inst_names, feat_names, target_names,
                      target_colors, P0, Ps, Ys[-1], Ys[:-1], repr_indices,
                      emb_of_Ys, cluster_ids)
    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./result/ces.json', 'w') as f:
        json.dump(result, f)
