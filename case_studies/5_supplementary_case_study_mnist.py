import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import scale
from umap import UMAP

import fealm.plot as fplot

import json

from output_formatter import dump_all

if __name__ == '__main__':
    # data loading and preparation
    # load data from https://www.openml.org/d/554
    # X, y = fetch_openml('mnist_784',
    #                     version=1,
    #                     return_X_y=True,
    #                     as_frame=False)
    X = np.load('./data/mnist_X.npy', allow_pickle=True)
    y = np.load('./data/mnist_y.npy', allow_pickle=True)
    y = y.astype(int)

    n_samples_each = 200
    target_idx = np.hstack((np.where(y == 4)[0][:n_samples_each],
                            np.where(y == 7)[0][:n_samples_each],
                            np.where(y == 9)[0][:n_samples_each]))
    X = X[target_idx, :]
    y = [0] * n_samples_each + [1] * n_samples_each + [2] * n_samples_each

    X = scale(X)

    n_neighbors = 10
    min_dist = 0.1

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X)
    fplot.plot_embeddings([Y], y)
    plt.show()

    # apply PCA
    from manopt_dr.core import gen_ldr
    from manopt_dr.predefined_func_generator import gen_cost_pca, gen_default_proj
    from pymanopt.manifolds import Stiefel, Grassmann
    from pymanopt.optimizers import TrustRegions

    PCA = gen_ldr(gen_cost_pca,
                  gen_default_proj,
                  manifold_generator=Grassmann,
                  optimizer=TrustRegions())

    n_components = 9
    pca = PCA(n_components, convergence_ratio=1e-8)
    X_pca = pca.fit_transform(X)
    P_pca = pca.M
    explained_var_ratios = X_pca.var(axis=0) / X.var(axis=0).sum()
    print('explained var ratio', explained_var_ratios)
    print('total explained var ratio', explained_var_ratios.sum())

    # from sklearn.decomposition import PCA as PCA2
    # pca2 = PCA2(n_components)
    # pca2.fit(X)
    # explained_var_ratios = pca2.explained_variance_ratio_
    # P_pca = pca2.components_.T
    # print('explained var ratio', explained_var_ratios)
    # print('total explained var ratio', explained_var_ratios.sum())

    fig = plt.figure(constrained_layout=True, figsize=(16, 7))
    subfigs = fig.subfigures(1, 2, width_ratios=[3, n_components // 3])
    axs0 = subfigs[0].subplots(1, 1)
    axs1 = subfigs[1].subplots(3, n_components // 3)
    for j in range(3):
        for k in range(n_components // 3):
            abs_max = np.abs(P_pca[:, j * (n_components // 3) + k]).max()
            axs1[j, k].imshow(P_pca[:, j * (n_components // 3) + k].reshape(
                (int(np.sqrt(P_pca.shape[0])), int(np.sqrt(P_pca.shape[0])))),
                              cmap='PRGn',
                              vmin=-abs_max,
                              vmax=abs_max)
            axs1[j, k].set(xticks=[])
            axs1[j, k].set(yticks=[])
            axs1[j, k].set(xlabel=None)
            axs1[j, k].set(ylabel=None)
            axs1[j, k].set_title(
                f'PC {j * (n_components // 3) + k + 1}\n(exp. var. ratio: {(explained_var_ratios[j * (n_components // 3) + k]):.3f})'
            )
    plt.show()

    feat_names = np.arange(0, X_pca.shape[1])
    inst_names = np.arange(0, X_pca.shape[0])
    target_names = ['f4', 'f7', 'f9']
    target_colors = np.array(['#507AA6', '#F08E39', '#5BA053'])

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    Y = umap.fit_transform(X_pca)
    fplot.plot_embeddings([Y], y)
    plt.show()

    from fealm.fealm import FEALM

    forms_to_settings = {
        'w': {
            'n_repeats': 10,
            'n_components': None,
            'pso_population_size': None,
            'pso_n_nonbest_solutions': 0,
            'pso_n_iterations': 3000,
            'lasso_coeff': -30,
            'ridge_coeff': 0
        },
        'p_wMv': {
            'n_repeats': 5,
            'n_components': 3,
            'pso_population_size': None,
            'pso_n_nonbest_solutions': 0,
            'pso_n_iterations': 5000,
            'lasso_coeff': 10,
            'ridge_coeff': -10
        }
    }

    Ps = []
    for form in forms_to_settings:
        fealm = FEALM(
            n_neighbors=n_neighbors,
            projection_form=form,
            n_components=forms_to_settings[form]['n_components'],
            n_repeats=forms_to_settings[form]['n_repeats'],
            pso_n_iterations=forms_to_settings[form]['pso_n_iterations'],
            pso_population_size=forms_to_settings[form]['pso_population_size'],
            pso_n_nonbest_solutions=forms_to_settings[form]
            ['pso_n_nonbest_solutions'],
            lasso_coeff=forms_to_settings[form]['lasso_coeff'],
            ridge_coeff=forms_to_settings[form]['ridge_coeff'])
        fealm = fealm.fit(X_pca)
        Ps += fealm.Ps

    P0 = np.diag([1] * X_pca.shape[1])
    cluster_result = fealm.find_representative_Ps(X_pca,
                                                  XP_dr_inst=umap,
                                                  Ps=Ps + [P0],
                                                  n_representatives=10)

    repr_indices = cluster_result['repr_indices']
    Ys = cluster_result['Ys']
    emb_of_Ys = cluster_result['emb_of_Ys']
    cluster_ids = cluster_result['cluster_ids']
    repr_Ys = cluster_result['repr_Ys']

    result = dump_all(X_pca, y, inst_names, feat_names, target_names,
                      target_colors, P0, Ps, Ys[-1], Ys[:-1], repr_indices,
                      emb_of_Ys, cluster_ids)

    fplot.plot_embeddings(repr_Ys, np.array(y))
    plt.show()

    with open('./result/mnist.json', 'w') as f:
        json.dump(result, f)

    # import os
    # outdir = './result/mnist'
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    #
    # import seaborn as sns
    # for i, Y in enumerate(Ys[:-1]):
    #     P = P_pca @ Ps[i]
    #
    #     abs_max = np.max(abs(P)) * 0.7
    #
    #     if i < forms_to_settings['w']['n_repeats']:
    #         n_pcs = n_components
    #     else:
    #         n_pcs = forms_to_settings['p_wMv']['n_components']
    #
    #     fig = plt.figure(constrained_layout=True, figsize=(13, 7))
    #     subfigs = fig.subfigures(1, 2, width_ratios=[3, max(n_pcs // 3, 2)])
    #     axs0 = subfigs[0].subplots(1, 1)
    #     axs1 = subfigs[1].subplots(3, max(n_pcs // 3, 2))
    #     data = {'x': Y[:, 0], 'y': Y[:, 1], 'label': y}
    #     sns.scatterplot(data=data,
    #                     x='x',
    #                     y='y',
    #                     hue='label',
    #                     ax=axs0,
    #                     palette='tab10',
    #                     zorder=2)
    #     axs0.set(xticks=[])
    #     axs0.set(yticks=[])
    #     axs0.set(xlabel=None)
    #     axs0.set(ylabel=None)
    #
    #     for j in range(3):
    #         for k in range(n_pcs // 3):
    #             pc = P[:, j * n_pcs // 3 + k]
    #             axs1[j, k].imshow(pc.reshape((int(np.sqrt(P_pca.shape[0])),
    #                                           int(np.sqrt(P_pca.shape[0])))),
    #                               cmap='coolwarm',
    #                               vmin=-abs_max,
    #                               vmax=abs_max)
    #             axs1[j, k].set(xticks=[])
    #             axs1[j, k].set(yticks=[])
    #             axs1[j, k].set(xlabel=None)
    #             axs1[j, k].set(ylabel=None)
    #             axs1[j, k].set_title(f'PC {j * (n_components // 3) + k + 1}')
    #     plt.tight_layout()
    #     plt.savefig(f'{outdir}/mnist{i}.png')
    #     # plt.show()
    # # plt.show()
    #
    # np.save(f'{outdir}/mnist_Ys.npy', Ys)
    # np.save(f'{outdir}/mnist_Ps_w.npy',
    #         Ps[0:forms_to_settings['w']['n_repeats']])
    # np.save(f'{outdir}/mnist_Ps_wMv.npy',
    #         Ps[forms_to_settings['w']['n_repeats']:])
    # np.save(f'{outdir}/mnist_P_pca.npy', P_pca)

    # import math
    # X_raw = np.load('./data/mnist_X.npy', allow_pickle=True)
    # X_raw = X_raw[target_idx, :]
    #
    # f4_sub = [
    #     1, 9, 11, 12, 13, 14, 19, 24, 25, 27, 28, 31, 33, 38, 39, 41, 42, 44,
    #     46, 50, 51, 52, 53, 55, 57, 62, 63, 65, 67, 71, 73, 85, 89, 94, 98, 99,
    #     100, 101, 107, 109, 111, 115, 117, 120, 122, 126, 131, 142, 146, 150,
    #     158, 167, 179, 189, 192, 196, 197, 319, 494, 507
    # ]
    # f4 = np.arange(100)
    # f4_oth = list(set(f4) - set(f4_sub))
    #
    # f7_sub = [
    #     200, 204, 207, 209, 211, 212, 217, 223, 224, 225, 226, 227, 228, 229,
    #     230, 231, 232, 233, 235, 236, 238, 239, 241, 242, 243, 247, 249, 252,
    #     256, 259, 261, 264, 265, 268, 271, 272, 273, 275, 276, 279, 281, 282,
    #     284, 285, 286, 287, 290, 291, 292, 293, 294, 296, 297, 300, 302, 304,
    #     308, 309, 310, 312, 314, 315, 324, 326, 328, 331, 333, 336, 337, 341,
    #     342, 343, 345, 346, 347, 348, 349, 350, 351, 357, 358, 361, 362, 363,
    #     365, 366, 367, 370, 371, 373, 374, 376, 377, 378, 379, 381, 382, 383,
    #     386, 387, 388, 389, 391, 393, 394, 395, 396, 397, 398, 399
    # ]
    # f7 = np.arange(200, 300)
    # f7_oth = list(set(f7) - set(f7_sub))
    #
    # fmix = [
    #     0, 2, 6, 18, 22, 30, 60, 72, 81, 82, 86, 91, 97, 103, 119, 127, 147,
    #     151, 153, 154, 176, 180, 198, 216, 248, 354, 364, 369, 390, 447, 479,
    #     488, 520, 555, 556, 566, 589, 599
    # ]
    # sub = fmix
    # n_cols = 19
    # n_rows = 4
    #
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    # for i, idx in enumerate(sub[:n_cols * n_rows]):
    #     ax = axes[i // n_cols, i % n_cols]
    #     ax.imshow(X_raw[idx].reshape(28, 28), cmap='gray')
    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         axes[i, j].set(xticks=[])
    #         axes[i, j].set(yticks=[])
    #         axes[i, j].set(xlabel=None)
    #         axes[i, j].set(ylabel=None)
    # plt.tight_layout()
    # plt.show()
