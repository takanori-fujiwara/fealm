import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

from umap import UMAP

from fealm.fealm import FEALM
import fealm.plot as fplot

import json

from output_formatter import dump_all

import pyreadstat  # please install via pip (pip3 install pyreadstat)
from ppic_meta import metainfo

if __name__ == '__main__':
    df, meta = pyreadstat.read_sav("./data/ppic/2018.10.24.release.sav")

    # covert code to binary, categorical, numeric numbers
    for col_key in df:
        df[col_key] = np.vectorize(metainfo[col_key]['code_num'])(df[col_key])

    # remove attributes that we are not interested in
    for col_key in df:
        if metainfo[col_key]['use'] == 0:
            del df[col_key]

    # we only want to compare democrat vs republican supporters
    # 1: Democrat, 2: Republican
    df.drop(np.where(df['q4a'] > 2)[0], axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # combine planned- and used-vote method
    df['q37a'][df['q37a'] == 999] = df['q37b'][df['q37a'] == 999]
    df.drop(['q37b'], axis=1, inplace=True)

    # also drop other non-important attributes to reduce # of nan
    # (e.g., there is a similar quesiton or related to the dropped cols)
    # 'q10: satisfy_senate_choice',
    # 'q12: congress_trump',
    # 'q15: importance_proposition6',
    # 'q17: imortance_proposition10',
    df.drop(['q10', 'q12', 'q15', 'q17'], axis=1, inplace=True)

    # remove attributes that have many nan values
    col_na_ratios = np.array(
        [np.sum(df[col_key] == 999) / df.shape[0] for col_key in df])
    df.drop(df.columns[np.where(col_na_ratios > 0.1)[0]], axis=1, inplace=True)

    # drop subjects who have NA in their answers
    row_na_ratios = np.array(
        [np.sum(row == 999) / df.shape[1] for index, row in df.iterrows()])
    df = df.iloc[row_na_ratios <= 0.0, :]
    df.reset_index(inplace=True, drop=True)

    # use q4a as label
    y = np.array(df['q4a'])
    y[y == 1] = 0  # democrat
    y[y == 2] = 1  # republican
    y_to_name = {0: 'Dem', 1: 'Rep'}
    target_names = np.array(['Dem', 'Rep'])
    target_colors = np.array(['#507AA6', '#DF585C'])

    df.drop(['q4a'], axis=1, inplace=True)
    X = np.array(df)
    inst_names = df.index

    feat_names = []
    name_to_q = {}
    for col_key in df.columns:
        name = col_key
        feat_names.append(name)
        name_to_q[name] = col_key
    feat_names = np.array(feat_names)

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    n_neighbors = 15
    min_dist = 0.1

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X)
    fplot.plot_embeddings([Y], y)
    plt.show()

    forms_to_settings = {
        'w': {
            'n_components': None,
            'pso_population_size': 200,
            'pso_n_nonbest_solutions': 10,
        },
        'p_wMv': {
            'n_components': 3,
            'pso_population_size': 500,
            'pso_n_nonbest_solutions': 20,
        }
    }

    Ps = []
    for form in forms_to_settings:
        fealm = FEALM(
            n_neighbors=n_neighbors,
            projection_form=form,
            n_components=forms_to_settings[form]['n_components'],
            n_repeats=5,
            pso_n_iterations=20,
            pso_population_size=forms_to_settings[form]['pso_population_size'],
            pso_n_nonbest_solutions=forms_to_settings[form]
            ['pso_n_nonbest_solutions'])
        fealm = fealm.fit(X)
        Ps += fealm.Ps

    P0 = np.diag([1] * X.shape[1])
    cluster_result = fealm.find_representative_Ps(X,
                                                  XP_dr_inst=umap,
                                                  Ps=fealm.Ps + [P0],
                                                  n_representatives=10)

    repr_indices = cluster_result['repr_indices']
    Ys = cluster_result['Ys']
    emb_of_Ys = cluster_result['emb_of_Ys']
    cluster_ids = cluster_result['cluster_ids']
    repr_Ys = cluster_result['repr_Ys']

    feat_names_with_short_desc = []
    for feat_name in feat_names:
        feat_names_with_short_desc.append(
            f'{feat_name}: {metainfo[feat_name]["short"]}')

    feat_names_capitlized = [name.capitalize() for name in feat_names]
    result = dump_all(np.array(df), y, inst_names, feat_names_capitlized,
                      target_names, target_colors, P0, Ps, Ys[-1], Ys[:-1],
                      repr_indices, emb_of_Ys, cluster_ids)

    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./results/ppic.json', 'w') as f:
        json.dump(result, f)
