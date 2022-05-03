import autograd.numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

from umap.umap_ import fuzzy_simplicial_set


def nearest_nbr_graph(X,
                      n_neighbors,
                      metric='minkowski',
                      p=2,
                      metric_params=None,
                      n_jobs=-1,
                      to_networx_graph=False):
    A = kneighbors_graph(X,
                         n_neighbors,
                         mode='connectivity',
                         metric=metric,
                         p=p,
                         metric_params=metric_params,
                         include_self=False,
                         n_jobs=n_jobs)
    # A is directed
    return nx.from_scipy_sparse_array(
        A, parallel_edges=False,
        create_using=nx.DiGraph) if to_networx_graph else A


def _euclidean_dist(X):
    i = np.ones(X.shape[0])[:, np.newaxis]
    G = X @ X.T
    g = np.diag(G)[:, np.newaxis]
    D = g @ i.T + i @ g.T - 2 * G

    return D


def fuzzy_nearest_nbr_graph(X,
                            n_neighbors,
                            metric='euclidean',
                            metric_kwds={},
                            random_state=None,
                            angular=False,
                            set_op_mix_ratio=1.0,
                            local_connectivity=1,
                            verbose=False,
                            to_networx_graph=False):
    A, _, _ = fuzzy_simplicial_set(X,
                                   n_neighbors=n_neighbors,
                                   random_state=random_state,
                                   metric=metric,
                                   metric_kwds=metric_kwds,
                                   knn_indices=None,
                                   knn_dists=None,
                                   angular=angular,
                                   set_op_mix_ratio=set_op_mix_ratio,
                                   local_connectivity=local_connectivity,
                                   apply_set_operations=True,
                                   verbose=verbose,
                                   return_dists=None)
    # A should be undirected (based on the paper)
    return nx.from_scipy_sparse_array(
        A, parallel_edges=False) if to_networx_graph else A
