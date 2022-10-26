#
# Note: Tested with netrd version 0.3.0
#
import time
import numpy as np
import pandas as pd
import networkx as nx

import netrd.distance as nd
import fealm.graph_func as gf
import fealm.graph_dissim as gd

from sklearn.neighbors import kneighbors_graph

if __name__ == '__main__':
    np.random.seed(0)

    k = 15  # UMAP's default
    m = 10
    ns = [50, 100, 200, 400, 800, 1600]
    n_prevs = [None, 50, 100, 200, 400, 800]
    n_repeats = 1000
    to_data_name = lambda n: f'./data/document_vec_n{n}_m{m}.npy'

    f_gr = lambda X: gf.nearest_nbr_graph(X, n_neighbors=k)
    f_gr_nx = lambda X: gf.nearest_nbr_graph(
        X, n_neighbors=k, to_networx_graph=True)

    f_gr_nx_undirected = lambda X: nx.from_scipy_sparse_array(
        kneighbors_graph(X, n_neighbors=k), parallel_edges=False)

    netrd_netlsd = nd.NetLSD()
    communicability_jsd = nd.CommunicabilityJSD()
    dk_series = nd.dkSeries()
    onion_divergence = nd.OnionDivergence()
    quantum_jsd = nd.QuantumJSD()
    degree_divergence = nd.DegreeDivergence()
    d_measure = nd.DMeasure()
    graph_diffusion = nd.GraphDiffusion()
    netsimile = nd.NetSimile()
    resistance_perturbation = nd.ResistancePerturbation()
    distributional_nbd = nd.DistributionalNBD()
    hamming = nd.Hamming()
    hamming_ipsen_mikhailov = nd.HammingIpsenMikhailov()
    ipsen_mikhailov = nd.IpsenMikhailov()
    jaccard_distance = nd.JaccardDistance()
    laplacian_spectral = nd.LaplacianSpectral()
    non_backtracking_spectral = nd.NonBacktrackingSpectral()
    polynomial_dissimilarity = nd.PolynomialDissimilarity()
    deltacon = nd.DeltaCon()
    frobenius = nd.Frobenius()
    portrait_divergence = nd.PortraitDivergence()

    d_gr_undirected = {
        # 1. Undirected, Unweighted
        'NetLSD (netrd)':
        lambda G1, G2: netrd_netlsd.dist(
            G1, G2, normalization=None, timescales=None),
        'NetLSD (ours)':
        lambda G1, G2, sig1, eigsh_v0: gd.netlsd(
            G1, G2, sig1=sig1, eigsh_v0=eigsh_v0),
        'Communicability JSD':
        communicability_jsd.dist,
        'dk Series':
        lambda G1, G2: dk_series.dist(G1, G2, d=2),
        'Onion Divergence':
        lambda G1, G2: onion_divergence.dist(G1, G2, dist='lccm'),
        'Quantum JSD':
        lambda G1, G2: quantum_jsd.dist(G1, G2, beta=0.1, q=None),
        # 2. Undirected
        'Degree Divergence':
        degree_divergence.dist,
        # 'd Measure': # d-measure doesn't work for nonconnected graph
        # lambda G1, G2: d_measure.dist(G1, G2, w1=0.45, w2=0.45, w3=0.1, niter=50),
        'Graph Diffusion':
        lambda G1, G2: graph_diffusion.dist(
            G1, G2, thresh=1e-08, resolution=1000),
        'NetSimile':
        netsimile.dist
        'Resistance Perturbation': # resistance_perturbation doesn't work for nonconnected graph
        lambda G1, G2: resistance_perturbation.dist(G1, G2, p=2)
    }

    d_gr_directed = {
        # 3. unweighted
        'Distributional NBD':
        lambda G1, G2: distributional_nbd.dist(G1,
                                               G2,
                                               sparse=False,
                                               shave=True,
                                               keep_evals=True,
                                               k=None,
                                               vector_distance='euclidean'),
        'Hamming':
        hamming.dist,
        'Hamming Ipsen Mikhailov':
        lambda G1, G2: hamming_ipsen_mikhailov.dist(
            G1, G2, combination_factor=1),
        'Ipsen Mikhailov':
        lambda G1, G2: ipsen_mikhailov.dist(G1, G2, hwhm=0.08),
        'Jaccard Distance':
        jaccard_distance.dist,
        'Laplacian Spectral':
        lambda G1, G2: laplacian_spectral.dist(G1,
                                               G2,
                                               normed=True,
                                               kernel='normal',
                                               hwhm=0.011775,
                                               measure='jensen-shannon',
                                               k=None,
                                               which='LM'),
        'Non-backtracking Spectral':
        lambda G1, G2: non_backtracking_spectral.dist(G1,
                                                      G2,
                                                      topk='automatic',
                                                      ignore_negative_evals=
                                                      True,
                                                      batch=50,
                                                      tol=1e-03),
        'Polynomial Dissimilarity':
        lambda G1, G2: polynomial_dissimilarity.dist(G1, G2, k=5, alpha=1),
        # 4. Any
        'DeltaCon (netrd)':
        lambda G1, G2: deltacon.dist(G1, G2, exact=True, g=None),
        'DeltaCon (ours)':
        lambda G1, G2, S1: gd.deltacon(G1, G2, S1),
        'Frobenius':
        frobenius.dist,
        'Portrait Divergence':
        lambda G1, G2: portrait_divergence.dist(
            G1, G2, bins=None, binedges=None)
    }

    d_gr = {'undirected': d_gr_undirected, 'directed': d_gr_directed}

    csv_path = './result/4_supplementary_cost_eval_dgraph.csv'
    file = open(csv_path, 'w')
    file.write('n,f_name,time\n')
    file.close()

    for n, n_prev in zip(ns, n_prevs):
        for directed_or_undirected in d_gr:
            d_gr_ = d_gr[directed_or_undirected]

            for f_name in d_gr_:
                # skip measures spent more than 60 seconds
                if n_prev:
                    df = pd.read_csv(
                        './result/4_supplementary_cost_eval_dgraph.csv')
                    prev_time = df.loc[(df['n'] == n_prev) &
                                       (df['f_name'] == f_name),
                                       df.columns == 'time']
                    if len(prev_time) == 0 or float(prev_time['time']) > 540:
                        print('skipped', f_name)
                        continue

                print(f_name)

                f = d_gr_[f_name]

                X = np.load(to_data_name(n))

                if f_name == 'NetLSD (ours)':
                    G1 = f_gr(X[:, :int(m / 2)])
                    G2 = f_gr(X[:, int(m / 2):])
                    if directed_or_undirected == 'undirected':
                        G1 = G1 + G1.T - G1 * G1.T
                        G2 = G2 + G2.T - G2 * G2.T
                    # utilize precomputation, which is effective when using FEALM
                    sig1 = gd._lsd_trace_signature(G1)
                    eigsh_v0 = np.random.rand(min(G1.shape))
                    kwargs = {
                        'G1': G1,
                        'G2': G2,
                        'sig1': sig1,
                        'eigsh_v0': eigsh_v0
                    }
                elif f_name == 'DeltaCon (ours)':
                    G1 = f_gr(X[:, :int(m / 2)])
                    G2 = f_gr(X[:, int(m / 2):])
                    S1 = gd._process_deltacon(G1, from_networkx_graph=False)
                    kwargs = {'G1': G1, 'G2': G2, 'S1': S1}
                else:
                    if directed_or_undirected == 'directed':
                        G1 = f_gr_nx(X[:, :int(m / 2)])
                        G2 = f_gr_nx(X[:, int(m / 2):])
                    if directed_or_undirected == 'undirected':
                        G1 = f_gr_nx_undirected(X[:, :int(m / 2)])
                        G2 = f_gr_nx_undirected(X[:, int(m / 2):])

                    kwargs = {'G1': G1, 'G2': G2}

                s = time.time()
                for i in range(n_repeats):
                    _ = f(**kwargs)
                e = time.time()

                file = open(csv_path, 'a')
                file.write(f'{n},{f_name},{e - s}\n')
                file.close()

                print(f'n{n} {f_name} {e - s}')

    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('./result/4_supplementary_cost_eval_dgraph.csv')

    # select only 10 representatives
    plotting_fnames = [
        'NetLSD (netrd)', 'NetLSD (ours)', 'Frobenius', 'DeltaCon (netrd)',
        'DeltaCon (ours)', 'Communicability JSD', 'NetSimile',
        'Laplacian Spectral', 'Portrait Divergence', 'Graph Diffusion'
    ]

    rows_selected = np.array([False] * df.shape[0])
    for f_name in plotting_fnames:
        rows_selected |= df['f_name'] == f_name

    df_selected = df.loc[rows_selected, :]

    data = pd.DataFrame({
        'n': df_selected['n'],
        'Completion Time (sec)': df_selected['time'],
        'Measure': df_selected['f_name']
    })

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=data, x='n', y='Completion Time (sec)', hue='Measure')
    plt.xlabel(r'$n$')
    plt.ylim([0, 540])
    plt.xlim([0, 1600])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('./result/4_supplementary_cost_eval_dgraph.pdf')
    plt.show()
