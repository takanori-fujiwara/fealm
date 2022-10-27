import os
import time
import numpy as np
import pandas as pd

import fealm.graph_func as gf
import fealm.graph_dissim as gd
from fealm.fealm import FEALM
from fealm.optimizer import AdaptiveNelderMead

if __name__ == '__main__':
    k = 15
    n = 800
    ms = [5, 10, 20]
    m_prime = 2
    n_evaluations = [100, 200, 400, 800, 1200, 1600, 2000]
    # randopt_population_size = None  # default is 10p + 1
    to_randopt_population_sizes = [
        lambda m, m_prime: (m * m_prime) + 1, lambda m, m_prime:
        (m * m_prime) * 10 + 1, lambda m, m_prime: (m * m_prime) * 20 + 1
    ]
    randopt_population_size_types = ['(p+1)', '(10p+1)', '(20p+1)']

    opt_n_eval = 5000  # optimal answer
    to_opt_randopt_population_size = lambda m, m_prime: (
        m * m_prime) * 50 + 1  # optimal answer
    n_jobs = -1

    form = 'no_constraint'  # or 'w'
    n_repeats = 10
    to_data_name = lambda n, m: f'./data/document_vec_n{n}_m{m}.npy'

    csv_path = './result/3_quality_eval.csv'

    file = open(csv_path, 'w')
    file.write(
        'form,m,n_eval,trial,opt_val,randopt_population_size,randopt_population_size_type\n'
    )
    file.close()

    for randopt_population_size_type, to_randopt_population_size in zip(
            randopt_population_size_types, to_randopt_population_sizes):
        for m in ms:
            X = np.load(to_data_name(n, m))
            for n_eval in n_evaluations:
                print(n_eval)
                # to avoid producing random solutions over n_eval
                randopt_population_size = to_randopt_population_size(
                    m, m_prime)
                if randopt_population_size > n_eval:
                    randopt_population_size = n_eval
                print('======', randopt_population_size)

                # set min_gradient_norm = 0 to avoid stopping by the convergence
                optimizer = AdaptiveNelderMead(
                    max_cost_evaluations=n_eval,
                    n_jobs=n_jobs,
                    randopt_population_size=randopt_population_size,
                    min_gradient_norm=0)

                fealm = FEALM(n_neighbors=k,
                              projection_form=form,
                              n_components=m_prime,
                              n_repeats=1,
                              optimizer=optimizer)

                for trial in range(n_repeats):
                    fealm = fealm.fit(X)
                    G1 = fealm.graph_func(X)
                    S1 = gd._shared_neighbor_sim(G1, k=k)
                    sig1 = gd._lsd_trace_signature(gd._to_undirected(G1))
                    Gs = [G1]
                    gd_preprocessed_data = [{'S1': S1, 'sig1': sig1}]
                    val = -fealm.opt._eval_cost(
                        X=X,
                        P=fealm.opt.P,
                        Gs=Gs,
                        gd_preprocessed_data=gd_preprocessed_data)

                    file = open(csv_path, 'a')
                    file.write(
                        f'{form},{m},{n_eval},{trial},{val},{randopt_population_size},{randopt_population_size_type}\n'
                    )
                    file.close()

                print(f'{form} {m} {n_eval} done')
                print()

    # optimal solutions
    for m in ms:
        X = np.load(to_data_name(n, m))
        # generate psuedo optimal solutions
        opt_randopt_population_size = to_opt_randopt_population_size(
            m, m_prime)

        optimizer = AdaptiveNelderMead(
            max_cost_evaluations=opt_n_eval,
            n_jobs=n_jobs,
            randopt_population_size=opt_randopt_population_size,
            min_gradient_norm=0,
            max_time=10800)

        fealm_optimal = FEALM(n_neighbors=k,
                              n_components=m_prime,
                              n_repeats=1,
                              optimizer=optimizer)

        for trial in range(n_repeats):
            fealm_optimal = fealm_optimal.fit(X)
            G1 = fealm_optimal.graph_func(X)
            S1 = gd._shared_neighbor_sim(G1, k=k)
            sig1 = gd._lsd_trace_signature(gd._to_undirected(G1))
            Gs = [G1]
            gd_preprocessed_data = [{'S1': S1, 'sig1': sig1}]
            opt_val = -fealm_optimal.opt._eval_cost(
                X=X,
                P=fealm_optimal.opt.P,
                Gs=Gs,
                gd_preprocessed_data=gd_preprocessed_data)

            file = open(csv_path, 'a')
            file.write(
                f'{form},{m},{opt_n_eval},{trial},{opt_val},{opt_randopt_population_size},(50p+1)\n'
            )
            file.close()
    # baseline solutions
    for m in ms:
        X = np.load(to_data_name(n, m))

        # same result with random selection of solution
        optimizer = AdaptiveNelderMead(max_cost_evaluations=1,
                                       randopt_population_size=1)

        fealm_optimal = FEALM(n_neighbors=k,
                              n_components=m_prime,
                              n_repeats=1,
                              optimizer=optimizer)

        for trial in range(n_repeats):
            fealm_optimal = fealm_optimal.fit(X)
            G1 = fealm_optimal.graph_func(X)
            S1 = gd._shared_neighbor_sim(G1, k=k)
            sig1 = gd._lsd_trace_signature(gd._to_undirected(G1))
            Gs = [G1]
            gd_preprocessed_data = [{'S1': S1, 'sig1': sig1}]
            opt_val = -fealm_optimal.opt._eval_cost(
                X=X,
                P=fealm_optimal.opt.P,
                Gs=Gs,
                gd_preprocessed_data=gd_preprocessed_data)

            file = open(csv_path, 'a')
            file.write(f'{form},{m},1,{trial},{opt_val},1,""\n')
            file.close()

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    csv_path = './result/3_quality_eval.csv'
    ms = [5, 10, 20]
    df = pd.read_csv(csv_path)
    df_mean = df.groupby(
        ['form', 'm', 'n_eval', 'randopt_population_size_type'],
        as_index=False).mean()

    df_mean['quality'] = df_mean['opt_val']

    for m in ms:
        optimal = float(
            df_mean[(df_mean['m'] == m)
                    & (df_mean['n_eval'] == opt_n_eval)]['opt_val'])
        # baseline = 0
        baseline = float(df_mean[(df_mean['m'] == m)
                                 & (df_mean['n_eval'] == 1)]['opt_val'])
        df_mean.iloc[df_mean['m'] == m, df_mean.columns ==
                     'quality'] = (df_mean['quality'][df_mean['m'] == m] -
                                   baseline) / (optimal - baseline)

    # exclude some of results, to simplify the figure
    df_mean = df_mean[df_mean['n_eval'] != 1]
    df_mean = df_mean[df_mean['n_eval'] != opt_n_eval]

    data = pd.DataFrame({
        'm':
        df_mean['m'],
        '# of Evaluations':
        df_mean['n_eval'],
        '# of Initial Solutions':
        df_mean['randopt_population_size_type'],
        'Relative Accuracy':
        df_mean['quality'],
        'Log m':
        np.log(df_mean['m']),
        'Log # of Iterations':
        np.log(df_mean['n_eval'])
    })

    newcolors = [[228, 26, 28], [77, 175, 74], [55, 126, 184]]
    newcolors = np.array(newcolors) / 255
    newcmp = ListedColormap(newcolors)

    palette = newcmp
    plt.figure(figsize=(2.75, 2.5))
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(p+1)'],
                     x='# of Evaluations',
                     y='Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls=':')
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(10p+1)'],
                     x='# of Evaluations',
                     y='Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls='--')
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(20p+1)'],
                     x='# of Evaluations',
                     y='Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls='-')
    g.get_legend().remove()
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    g.xaxis.set_ticks_position('bottom')
    g.yaxis.set_ticks_position('left')
    plt.ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('./result/3_quality_eval_mean.pdf')
    plt.show()

    ### Standard deviation of qualities
    df['quality'] = df['opt_val']

    df_mean = df.groupby(
        ['form', 'm', 'n_eval', 'randopt_population_size_type'],
        as_index=False).mean()

    df_mean['quality'] = df_mean['opt_val']

    for m in ms:
        optimal = float(
            df_mean[(df_mean['m'] == m)
                    & (df_mean['n_eval'] == opt_n_eval)]['opt_val'])
        # baseline = 0
        baseline = float(df_mean[(df_mean['m'] == m)
                                 & (df_mean['n_eval'] == 1)]['opt_val'])
        df.iloc[df['m'] == m,
                df.columns == 'quality'] = (df['quality'][df['m'] == m] -
                                            baseline) / (optimal - baseline)

    df_std = df.groupby(
        ['form', 'm', 'n_eval', 'randopt_population_size_type'],
        as_index=False).std()

    # exclude some of results, to simplify the figure
    df_std = df_std[df_std['n_eval'] != opt_n_eval]
    df_std = df_std[df_std['n_eval'] != 1]

    data = pd.DataFrame({
        'm':
        df_std['m'],
        '# of Evaluations':
        df_std['n_eval'],
        '# of Initial Solutions':
        df_mean['randopt_population_size_type'],
        'SD of Relative Accuracy':
        df_std['quality'],
        'SD of Objective Values':
        df_std['opt_val'],
    })

    palette = newcmp
    plt.figure(figsize=(4, 3))
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(p+1)'],
                     x='# of Evaluations',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls=':')
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(10p+1)'],
                     x='# of Evaluations',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls='--')
    g = sns.lineplot(data=data[data['# of Initial Solutions'] == '(20p+1)'],
                     x='# of Evaluations',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls='-')
    g.get_legend().remove()
    plt.tight_layout()
    plt.savefig('./result/3_quality_eval_std.pdf')
    plt.show()
