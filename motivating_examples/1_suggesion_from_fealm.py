import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP

import fealm.graph_dist as gd
import fealm.plot as fplot
from fealm.fealm import FEALM

if __name__ == '__main__':
    ### choose dataset # TODO: test non-scaling one
    target_dataset = 'two_spheres_with_three_class_attr'  # Dataset 2
    # target_dataset = 'two_spheres_with_three_class_attr_disturbance'  # Dataset3

    ### choose d_Gr
    graph_dissim_measure = 'nsd'
    # graph_dissim_measure = 'netlsd'
    # graph_dissim_measure = 'nd'

    X = np.load(f'./data/{target_dataset}.npy')
    y = np.load('./data/two_sphere_label.npy')
    y2 = np.load('./data/three_class_label.npy')

    if target_dataset == 'two_spheres_with_three_class_attr':
        form_and_sizes = {
            'w': {
                'population_size': 500,
                'n_results': 10,
                'result_selection': 'P'
            }
        }
        n_repeats = 20
        pso_niter = 5
    elif target_dataset == 'two_spheres_with_three_class_attr_disturbance':
        form_and_sizes = {
            'p_wMv': {
                'population_size': 1000,
                'n_results': 20,
                'result_selection': 'P'
            }
        }
        n_repeats = 30
        pso_niter = 5

    n_neighbors = 15
    min_dist = 0.1
    n_components = 3
    pso_maxtime = 3000
    n_representatives = 30

    if graph_dissim_measure == 'nsd':
        graph_dist = lambda G1, G2, S1=None, sig1=None: gd.nsd(
            G1,
            G2,
            beta=1,
            fixed_degree=n_neighbors,
            S1=S1,
            sig1=sig1,
            snn_nhops=1,
            snn_symmetrize=False,
            take_log_for_lsd=True)
    elif graph_dissim_measure == 'netlsd':
        graph_dist = lambda G1, G2, S1=None, sig1=None: gd.netlsd(
            G1, G2, sig1=sig1)
    elif graph_dissim_measure == 'nd':
        graph_dist = lambda G1, G2, S1=None, sig1=None: gd.snn_dissim(
            G1,
            G2,
            S1=S1,
            fixed_degree=n_neighbors,
            symmetrize=False,
            n_hops=3)

    fealm = FEALM(n_neighbors=n_neighbors,
                  n_components=n_components,
                  n_repeats=n_repeats,
                  pso_maxtime=pso_maxtime,
                  pso_niter=pso_niter,
                  form_and_sizes=form_and_sizes,
                  graph_dist=graph_dist)

    fealm = fealm.fit(X)
    Ps = fealm.Ps
    best_P_indices = fealm.best_P_indices
    print('fitting is done')

    P0 = np.diag([1] * X.shape[1])
    repr_Ps_info = fealm.find_representative_Ps(
        X,
        X2Y_dr_inst=UMAP(n_components=2,
                         n_neighbors=n_neighbors,
                         min_dist=min_dist),
        Ps=fealm.Ps + [P0],
        n_representatives=n_representatives,
        clustering_on_emb_of_Ys=True)

    repr_Ys = repr_Ps_info['repr_Ys']
    Ys = repr_Ps_info['Ys']

    # plot representative results
    for i in range(len(repr_Ys) // 5):
        fplot.plot_embeddings(repr_Ys[i * 5:(i + 1) * 5], y + 1)
        plt.savefig(f'./result/{target_dataset}_repr{i}_y1.png')
        plt.show()
        plt.close()
        fplot.plot_embeddings(repr_Ys[i * 5:(i + 1) * 5], y2 + 1, 'Dark2_r')
        plt.savefig(f'./result/{target_dataset}_repr{i}_y2.png')
        plt.show()
        plt.close()

    # plot all results
    for i in range(len(Ys) // 5):
        fplot.plot_embeddings(Ys[i * 5:(i + 1) * 5], y + 1)
        plt.savefig(f'./result/{target_dataset}_{i}_y1.png')
        # plt.show()
        plt.close()
        fplot.plot_embeddings(Ys[i * 5:(i + 1) * 5], y2 + 1, 'Dark2_r')
        plt.savefig(f'./result/{target_dataset}_{i}_y2.png')
        # plt.show()
        plt.close()
