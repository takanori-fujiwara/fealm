import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale
from umap import UMAP

from fealm.fealm import FEALM
import fealm.plot as fplot

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

    fealm = FEALM(n_neighbors=n_neighbors, projection_form='w', n_repeats=5)

    fealm = fealm.fit(X)
    Ps = fealm.Ps
    best_P_indices = fealm.best_P_indices

    P0 = np.diag([1] * X.shape[1])
    cluster_result = fealm.find_representative_Ps(X,
                                                  XP_dr_inst=umap,
                                                  Ps=Ps + [P0],
                                                  n_representatives=10,
                                                  clustering_on_emb_of_Ys=True)

    fplot.plot_embeddings(cluster_result['repr_Ys'], np.array(y))
    plt.show()
