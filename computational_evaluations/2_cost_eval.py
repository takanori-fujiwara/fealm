import time
import numpy as np
import pandas as pd

if __name__ == '__main__':
    k = 15  # UMAP's default
    m = 10
    q = 50
    ns = [50, 100, 200, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
    n_repeats = 1000
    to_data_name = lambda n: f'./data/document_vec_n{n}_m{m}.npy'

    from umap import UMAP
    umap = UMAP(n_components=2, n_neighbors=k)
    f_dr = umap.fit_transform

    import fealm.graph_func as gf
    import fealm.graph_dissim as gd

    f_gr = lambda X: gf.nearest_nbr_graph(X, n_neighbors=k)

    # In PSO, Gi+1/G1 is usually fixed and S1 and sig1 can be computed in advance
    d_nd = lambda G1, G2, S1, sig1: gd.snn_dissim(
        G1, G2, S1=S1, fixed_degree=k)

    d_sd = lambda G1, G2, S1, sig1: gd.netlsd(
        G1, G2, sig1=sig1, n_eigvals=(int(q / 2), int(q / 2)))

    beta = 1
    d_nsd = lambda G1, G2, S1, sig1: d_nd(G1, G2, S1, sig1)**beta + np.log(
        1 + d_sd(G1, G2, S1, sig1))

    # n_iter and walk_ratio follow Jeon et al.'s default
    d_snc = lambda G1, G2, S1, sig1: gd.snc_dissim(
        G1, G2, S1=S1, n_iter=200, walk_ratio=0.4)

    f_names_and_fs = {
        'f_dr': f_dr,
        'f_gr': f_gr,
        'd_nd': d_nd,
        'd_sd': d_sd,
        'd_nsd': d_nsd,
        'd_snc': d_snc
    }

    result = []
    for n in ns:
        for f_name in f_names_and_fs:
            print(f'n{n} {f_name}')
            f = f_names_and_fs[f_name]
            X = np.load(to_data_name(n))
            kwargs = {'X': X}

            # f_dr and d_snc are much slower than others to run even for n=50
            # thus, we do not evaluate for n >= 100
            if f_name in ['f_dr', 'd_snc'] and n >= 100:
                continue

            if f_name not in ['f_dr', 'f_gr']:
                G1 = f_gr(X[:, :int(m / 2)])
                G2 = f_gr(X[:, int(m / 2):])
                S1 = gd._shared_neighbor_sim(G1, k=k)
                sig1 = gd._lsd_trace_signature(G1,
                                               n_eigvals=(int(q / 2),
                                                          int(q / 2)))
                kwargs = {'G1': G1, 'G2': G2, 'S1': S1, 'sig1': sig1}

            s = time.time()
            for i in range(n_repeats):
                _ = f(**kwargs)
            e = time.time()

            result.append({'n': n, 'f_name': f_name, 'time': e - s})
            print(f'n{n} {f_name} {e - s}')

    m_prime = 2
    n_jobs = 4

    from fealm.fealm import FEALM
    from fealm.optimizer import AdaptiveNelderMead

    # set min_gradient_norm = 0 to avoid stopping by the convergence
    optimizer = AdaptiveNelderMead(max_cost_evaluations=n_repeats,
                                   n_jobs=n_jobs,
                                   min_gradient_norm=0)
    for n in ns:
        X = np.load(to_data_name(n))
        fealm = FEALM(n_neighbors=k,
                      projection_form='no_constraint',
                      n_components=m_prime,
                      n_repeats=1,
                      optimizer=optimizer)

        s = time.time()
        fealm = fealm.fit(X)
        e = time.time()
        result.append({'n': n, 'f_name': 'FEALM-UMAP', 'time': e - s})
        print(f'n{n} FEALM-UMAP {e - s}')

    pd.DataFrame(result).to_csv('./result/2_cost_eval.csv', index=False)

    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('./result/2_cost_eval.csv')

    # remove f_dr and d_snc (much slower than others)
    df = df[df['f_name'] != 'f_dr']
    df = df[df['f_name'] != 'd_snc']

    df.loc[df['f_name'] == 'FEALM-UMAP', df.columns == 'f_name'] = 'FEALM'
    df.loc[df['f_name'] == 'f_gr', df.columns == 'f_name'] = r'$k$-NN'
    df.loc[df['f_name'] == 'd_nd', df.columns == 'f_name'] = r'$d_\mathrm{ND}$'
    df.loc[df['f_name'] == 'd_sd', df.columns == 'f_name'] = r'$d_\mathrm{SD}$'
    df.loc[df['f_name'] == 'd_nsd',
           df.columns == 'f_name'] = r'$d_\mathrm{NSD}$'
    hue_order = [
        'FEALM', r'$d_\mathrm{NSD}$', r'$d_\mathrm{ND}$', r'$d_\mathrm{SD}$',
        r'$k$-NN'
    ]

    data = pd.DataFrame({
        'n': df['n'],
        'Completion Time (sec)': df['time'],
        'Function': df['f_name']
    })

    # plt.figure(figsize=(4, 2.5))
    fig, ax = plt.subplots(figsize=(2.75, 2.5))
    sns.lineplot(data=data,
                 x='n',
                 y='Completion Time (sec)',
                 hue='Function',
                 hue_order=hue_order)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    plt.xlabel(r'$n$')
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('./result/2_cost_eval.pdf')
    plt.show()
