import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import preprocessing
from umap import UMAP

from fealm.fealm import FEALM
import fealm.plot as fplot

import json

from output_formatter import dump_all

if __name__ == '__main__':
    tableau10 = {
        'teal': '#78B7B2',
        'blue': '#507AA6',
        'orange': '#F08E39',
        'red': '#DF585C',
        'green': '#5BA053',
        'purple': '#AF7BA1',
        'yellow': '#ECC854',
        'brown': '#9A7460',
        'pink': '#FD9EA9',
        'gray': '#BAB0AC'
    }

    dataset = datasets.load_wine()
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    feat_names = dataset.feature_names
    inst_names = np.arange(0, X.shape[1])
    target_colors = np.array(
        [tableau10['blue'], tableau10['orange'], tableau10['green']])

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    n_neighbors = 15
    min_dist = 0.1

    Y = UMAP(n_components=2, n_neighbors=n_neighbors,
             min_dist=min_dist).fit_transform(X)
    fplot.plot_embeddings([Y], y)
    plt.show()

    n_components = 3
    form_and_sizes = {
        'w': {
            'population_size': 500,
            'n_results': 20,
        },
        'p_wMv': {
            'population_size': 1000,
            'n_results': 20,
        }
    }
    n_repeats = 10
    pso_niter = 20
    fealm = FEALM(n_neighbors=n_neighbors,
                  n_components=n_components,
                  n_repeats=n_repeats,
                  pso_maxtime=3000,
                  pso_niter=pso_niter,
                  form_and_sizes=form_and_sizes)

    fealm = fealm.fit(X)
    Ps = fealm.Ps
    best_P_indices = fealm.best_P_indices

    P0 = np.diag([1] * X.shape[1])
    cluster_result = fealm.find_representative_Ps(
        X,
        X2Y_dr_inst=UMAP(n_components=2,
                         n_neighbors=n_neighbors,
                         min_dist=min_dist),
        Ps=fealm.Ps + [P0],
        n_representatives=10,
        Ys_dr_kwargs={
            'n_components': 2,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
        },
        clustering_on_emb_of_Ys=True)

    closest_indices = cluster_result['closest_Y_indices']
    Ys = cluster_result['Ys']
    emb_of_Ys = cluster_result['emb_of_Ys']
    cluster_ids = cluster_result['cluster_ids']
    repr_Ys = cluster_result['repr_Ys']

    result = dump_all(dataset.data, y, inst_names, feat_names, target_names,
                      target_colors, P0, Ps, Ys[-1], Ys[:-1], closest_indices,
                      emb_of_Ys, cluster_ids)

    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./result/wine.json', 'w') as f:
        json.dump(result, f)
