import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale
from umap import UMAP

from fealm.fealm import FEALM
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

    forms_to_settings = {
        'w': {
            'n_components': None,
            'pso_population_size': 500,
            'pso_n_nonbest_solutions': 20,
        },
        'p_wMv': {
            'n_components': 3,
            'pso_population_size': 1000,
            'pso_n_nonbest_solutions': 20,
        }
    }

    Ps = []
    for form in forms_to_settings:
        fealm = FEALM(
            n_neighbors=n_neighbors,
            projection_form=form,
            n_components=forms_to_settings[form]['n_components'],
            n_repeats=10,
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

    result = dump_all(dataset.data, y, inst_names, feat_names, target_names,
                      target_colors, P0, Ps, Ys[-1], Ys[:-1], repr_indices,
                      emb_of_Ys, cluster_ids)

    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./result/wine.json', 'w') as f:
        json.dump(result, f)
