import time
import numpy as np
import pandas as pd

import fealm.graph_func as gf
import fealm.graph_dist as gd

if __name__ == '__main__':
    k = 15
    m = 10
    m_prime = int(m / 2)
    n = 200
    n_samplings = 500
    to_data_name = lambda n: f'./data/document_vec_n{n}_m{m}.npy'

    f_gr = lambda X: gf.nearest_nbr_graph(X, n_neighbors=k)
    d_nd = lambda G1, G2: gd.snn_dissim(G1, G2, fixed_degree=k)
    d_snc = lambda G1, G2: gd.snc_dissim(G1, G2, n_iter=200, walk_ratio=0.4)

    # generate P mimicing no constraint condition
    man_rand = lambda: np.random.randn(m, m_prime)

    nd_results = []
    snc_results = []

    X = np.load(to_data_name(n))

    G1 = f_gr(X)
    Ps = [man_rand() for _ in range(n_samplings)]
    G2s = [f_gr(X @ P) for P in Ps]

    nd_results = [d_nd(G1, G2) for G2 in G2s]

    set_of_snc_results = []
    s = time.time()
    for i in range(50):
        print(i)
        snc_results = [d_snc(G1, G2) for G2 in G2s]
        set_of_snc_results.append(snc_results)
        print(time.time() - s)
    mean_snc_results = np.array(set_of_snc_results).mean(axis=0)

    pd.DataFrame({
        'nd': nd_results,
        'snc': mean_snc_results
    }).to_csv('./result/2_nd_vs_snc.csv', index=False)

    np.save('./result/2_set_of_snc_results.npy', set_of_snc_results)
    # np.load('set_of_snc_results.npy')

    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    df = pd.read_csv('./result/2_nd_vs_snc.csv').iloc[:500, :]
    print('pearsonr:', pearsonr(df['snc'], df['nd']))
    print('spearmanr:', spearmanr(df['snc'], df['nd']))
    # both over 0.7: strong correlation

    data = pd.DataFrame({'SnC': df['snc'], 'ND': df['nd']})

    sns.lmplot(data=data,
               x='SnC',
               y='ND',
               scatter_kws={
                   "s": 20,
                   "alpha": 1,
                   'color': 'gray'
               },
               line_kws={'color': 'tab:pink'},
               ci=None,
               height=2.5,
               aspect=1.6)
    plt.xlabel(r'$d_\mathrm{SnC}$')
    plt.ylabel(r'$d_\mathrm{ND}$')
    plt.tight_layout()
    plt.savefig('./result/2_nd_vs_snc.pdf')
    plt.show()
