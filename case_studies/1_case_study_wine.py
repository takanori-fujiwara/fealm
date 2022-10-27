import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale
from umap import UMAP

import fealm.plot as fplot

import json

from output_formatter import dump_all

if __name__ == '__main__':
    dataset = datasets.load_wine()
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    feat_names = dataset.feature_names
    inst_names = np.arange(0, X.shape[1])
    target_colors = np.array(['#507AA6', '#F08E39', '#5BA053'])

    X = scale(X)

    n_neighbors = 15
    min_dist = 0.1

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X)
    fplot.plot_embeddings([Y], y)
    plt.show()

    from fealm.fealm import FEALM
    from fealm.optimizer import AdaptiveNelderMead

    forms_to_settings = {
        'w': {
            'n_repeats': 30,
            'n_components': None,
            'max_cost_evaluations': 3000,
            'lasso_coeff': -10,
            'ridge_coeff': 0,
        },
        'p_wMv': {
            'n_repeats': 10,
            'n_components': 3,
            'max_cost_evaluations': 5000,
            'lasso_coeff': 10,
            'ridge_coeff': 0,
        }
    }

    Ps = []
    for form in forms_to_settings:
        optimizer = AdaptiveNelderMead(
            max_cost_evaluations=forms_to_settings[form]
            ['max_cost_evaluations'])

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
    cluster_result = fealm.find_representative_Ps(X,
                                                  XP_dr_inst=umap,
                                                  Ps=Ps + [P0],
                                                  n_representatives=10)

    repr_indices = cluster_result['repr_indices']
    Ys = cluster_result['Ys']
    emb_of_Ys = cluster_result['emb_of_Ys']
    cluster_ids = cluster_result['cluster_ids']
    repr_Ys = cluster_result['repr_Ys']

    result = dump_all(dataset.data, y, inst_names, feat_names, target_names,
                      target_colors, P0, Ps, Ys[-1], Ys[:-1], repr_indices,
                      emb_of_Ys, cluster_ids)

    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./result/wine.json', 'w') as f:
        json.dump(result, f)
