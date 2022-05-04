import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP

import fealm.graph_dissim as gd
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
        projection_form = 'w'
        n_components = None

        n_repeats = 20
        pso_n_nonbest_solutions = 10
        pso_population_size = 500
        pso_n_iterations = 5
    elif target_dataset == 'two_spheres_with_three_class_attr_disturbance':
        projection_form = 'p_wMv'
        n_components = 3

        n_repeats = 30
        pso_n_nonbest_solutions = 20
        pso_population_size = 1000
        pso_n_iterations = 5

    n_neighbors = 15
    min_dist = 0.1
    n_representatives = 30

    if graph_dissim_measure == 'nsd':
        graph_dissim = lambda G1, G2, S1=None, sig1=None: gd.nsd(
            G1, G2, beta=1, fixed_degree=n_neighbors, S1=S1, sig1=sig1)
    elif graph_dissim_measure == 'netlsd':
        graph_dissim = lambda G1, G2, S1=None, sig1=None: gd.netlsd(
            G1, G2, sig1=sig1)
    elif graph_dissim_measure == 'nd':
        graph_dissim = lambda G1, G2, S1=None, sig1=None: gd.snn_dissim(
            G1, G2, S1=S1, fixed_degree=n_neighbors)

    fealm = FEALM(n_neighbors=n_neighbors,
                  n_repeats=n_repeats,
                  projection_form=projection_form,
                  n_components=n_components,
                  pso_n_iterations=pso_n_iterations,
                  graph_dissim=graph_dissim)

    fealm = fealm.fit(X)
    Ps = fealm.Ps
    best_P_indices = fealm.best_P_indices
    print('fitting is done')

    P0 = np.diag([1] * X.shape[1])
    repr_Ps_info = fealm.find_representative_Ps(
        X,
        XP_dr_inst=UMAP(n_neighbors=n_neighbors, min_dist=min_dist),
        Ps=Ps + [P0],
        n_representatives=n_representatives)

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
