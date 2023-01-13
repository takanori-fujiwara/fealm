import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import scale
from umap import UMAP

import fealm.graph_dissim as gd

if __name__ == '__main__':
    target_dataset = 'two_spheres_with_three_class_attr'
    # target_dataset = 'two_spheres_with_three_class_attr_disturbance'
    # target_dataset = 'wine'

    if target_dataset == 'wine':
        X = scale(datasets.load_wine().data)
    else:
        X = np.load(f'../motivating_examples/data/{target_dataset}.npy')

    n_neighbors = 15
    min_dist = 0.1
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X)

    from fealm.fealm import FEALM
    from fealm.optimizer import AdaptiveNelderMead

    if target_dataset == 'two_spheres_with_three_class_attr':
        beta = 1
        projection_form = 'w'
        n_components = None
        lasso_coeff = -100
        ridge_coeff = 0
    elif target_dataset == 'two_spheres_with_three_class_attr_disturbance':
        beta = 1
        projection_form = 'p_wMv'
        n_components = 3
        lasso_coeff = 50
        ridge_coeff = -100
    elif target_dataset == 'wine':
        beta = 1
        projection_form = 'w'
        n_components = None
        lasso_coeff = -10
        ridge_coeff = 0

    graph_dissim = lambda G1, G2, S1=None, sig1=None: gd.nsd(
        G1, G2, beta=beta, fixed_degree=n_neighbors, S1=S1, sig1=sig1)

    optimizer = AdaptiveNelderMead(max_cost_evaluations=5000)
    fealm = FEALM(n_neighbors=n_neighbors,
                  n_repeats=1,
                  projection_form=projection_form,
                  n_components=n_components,
                  graph_dissim=graph_dissim,
                  lasso_coeff=lasso_coeff,
                  ridge_coeff=ridge_coeff,
                  optimizer=optimizer)

    n_repeats = 50
    Gs = [fealm.graph_func(X)]
    Ps = []
    vals = []
    for i in range(n_repeats):
        fealm = fealm.fit(X, Gs=Gs)
        new_P = fealm.Ps[-1]
        new_G = fealm.graph_func(X @ new_P)
        val = -fealm.opt._eval_cost(X=X, P=new_P, Gs=Gs)

        Gs.append(new_G)
        Ps.append(new_P)
        vals.append(val)
        print(len(vals))
        print(vals)

    plt.scatter(np.arange(len(vals)), vals)
    plt.show()
