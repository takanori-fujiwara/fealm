import os
import time
import numpy as np
import pandas as pd

import fealm.graph_func as gf
import fealm.graph_dissim as gd
from fealm.fealm import FEALM

if __name__ == '__main__':
    k = 15
    n = 400
    ms = [5, 10, 20]
    n_iters = [5, 10, 20]
    population_sizes = [100, 200, 400, 800, 1600]
    opt_n_iter = 40  # optimal answer
    opt_population_size = 3200  # optimal answer
    n_jobs = -1

    # forms = ['w', 'no_constraint']
    forms = ['no_constraint']
    n_repeats = 10
    to_data_name = lambda n, m: f'./data/document_vec_n{n}_m{m}.npy'

    csv_path = './result/4_fealm_quality_eval.csv'
    file = open(csv_path, 'w')
    file.write('form,m,n_iter,population_size,trial,opt_val\n')
    file.close()

    for form in forms:
        for m in ms:
            X = np.load(to_data_name(n, m))
            m_prime = int(m / 2)

            for n_iter in n_iters:
                for population_size in population_sizes:
                    fealm = FEALM(n_neighbors=k,
                                  projection_form=form,
                                  n_components=m_prime,
                                  n_repeats=1,
                                  pso_n_iterations=n_iter,
                                  pso_n_jobs=n_jobs,
                                  pso_n_nonbest_solutions=0,
                                  pso_population_size=population_size)

                    for trial in range(n_repeats):
                        fealm = fealm.fit(X)
                        G1 = fealm.graph_func(X)
                        S1 = gd._shared_neighbor_sim(G1, k=k)
                        sig1 = gd._lsd_trace_signature(gd._to_undirected(G1))
                        Gs = [G1]
                        gd_preprocessed_data = [{'S1': S1, 'sig1': sig1}]
                        opt_val = 1 / fealm.opt._eval_cost(
                            X=X,
                            P=fealm.opt.P,
                            Gs=Gs,
                            gd_preprocessed_data=gd_preprocessed_data)

                        file = open(csv_path, 'a')
                        file.write(
                            f'{form},{m},{n_iter},{population_size},{trial},{opt_val}\n'
                        )
                        file.close()

                    print(f'{form} {m} {n_iter} {population_size} done')
                    print()

            # generate psuedo optimal solutions
            fealm_optimal = FEALM(n_neighbors=k,
                                  n_components=m_prime,
                                  n_repeats=1,
                                  pso_maxtime=7200,
                                  pso_n_iterations=opt_n_iter,
                                  pso_population_size=opt_population_size,
                                  pso_n_nonbest_solutions=0,
                                  pso_njobs=n_jobs)
            for trial in range(n_repeats):
                fealm_optimal = fealm_optimal.fit(X)
                G1 = fealm_optimal.graph_func(X)
                S1 = gd._shared_neighbor_sim(G1, k=k)
                sig1 = gd._lsd_trace_signature(gd._to_undirected(G1))
                Gs = [G1]
                gd_preprocessed_data = [{'S1': S1, 'sig1': sig1}]
                opt_val = 1 / fealm_optimal.opt._eval_cost(
                    X=X,
                    P=fealm_optimal.opt.P,
                    Gs=Gs,
                    gd_preprocessed_data=gd_preprocessed_data)

                file = open(csv_path, 'a')
                file.write(
                    f'{form},{m},{opt_n_iter},{opt_population_size},{trial},{opt_val}\n'
                )
                file.close()

    import seaborn as sns
    import matplotlib.pyplot as plt

    ms = [5, 10, 20]
    df = pd.read_csv(csv_path)
    df_mean = df.groupby(['form', 'm', 'n_iter', 'population_size'],
                         as_index=False).mean()

    df_mean['quality'] = df_mean['opt_val']

    optimal_niter = 40
    optimal_population_size = 3200
    for m in ms:
        optimal = float(df_mean[(df_mean['m'] == m)
                                & (df_mean['n_iter'] == optimal_niter) &
                                (df_mean['population_size']
                                 == optimal_population_size)]['opt_val'])
        baseline = 0
        df_mean.loc[df_mean['m'] == m, df_mean.columns ==
                    'quality'] = (df_mean['quality'][df_mean['m'] == m] -
                                  baseline) / (optimal - baseline)

    # exclude some of results, to simplify the figure
    df_mean = df_mean[df_mean['m'] != optimal_niter]
    df_mean = df_mean[df_mean['n_iter'] != optimal_niter]
    df_mean = df_mean[df_mean['population_size'] != optimal_population_size]

    data = pd.DataFrame({
        '# of Particles': df_mean['population_size'],
        'm': df_mean['m'],
        '# of Iterations': df_mean['n_iter'],
        'Mean Relative Accuracy': df_mean['quality'],
        'Log m': np.log(df_mean['m']),
        'Log # of Iterations': np.log(df_mean['n_iter'])
    })

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    newcolors = [[228, 26, 28], [77, 175, 74], [55, 126, 184]]
    newcolors = np.array(newcolors) / 255
    newcmp = ListedColormap(newcolors)

    palette = newcmp
    plt.figure(figsize=(4, 3))
    g = sns.lineplot(data=data[data['# of Iterations'] == 5],
                     x='# of Particles',
                     y='Mean Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls=':')
    g = sns.lineplot(data=data[data['# of Iterations'] == 10],
                     x='# of Particles',
                     y='Mean Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls='--')
    g = sns.lineplot(data=data[data['# of Iterations'] == 20],
                     x='# of Particles',
                     y='Mean Relative Accuracy',
                     hue='m',
                     palette=palette,
                     ls='-')
    g.get_legend().remove()
    plt.tight_layout()
    plt.savefig('./result/4_fealm_quality_eval.pdf')
    plt.show()

    ### Standard deviation of qualities
    df['quality'] = df['opt_val']

    df_mean = df.groupby(['form', 'm', 'n_iter', 'population_size'],
                         as_index=False).mean()

    df_mean['quality'] = df_mean['opt_val']

    optimal_niter = 40
    optimal_population_size = 3200
    for m in ms:
        optimal = float(df_mean[(df_mean['m'] == m)
                                & (df_mean['n_iter'] == optimal_niter) &
                                (df_mean['population_size']
                                 == optimal_population_size)]['opt_val'])
        baseline = 0
        df_mean.loc[df_mean['m'] == m, df_mean.columns ==
                    'quality'] = (df_mean['quality'][df_mean['m'] == m] -
                                  baseline) / (optimal - baseline)

    for m in ms:
        optimal = float(df_mean[(df_mean['m'] == m)
                                & (df_mean['n_iter'] == optimal_niter) &
                                (df_mean['population_size']
                                 == optimal_population_size)]['opt_val'])
        baseline = 0
        df.loc[df['m'] == m,
               df.columns == 'quality'] = (df['quality'][df['m'] == m] -
                                           baseline) / (optimal - baseline)

    df_std = df.groupby(['form', 'm', 'n_iter', 'population_size'],
                        as_index=False).std()

    # exclude some of results, to simplify the figure
    df_std = df_std[df_std['m'] != optimal_niter]
    df_std = df_std[df_std['n_iter'] != optimal_niter]
    df_std = df_std[df_std['population_size'] != optimal_population_size]

    data = pd.DataFrame({
        '# of Particles': df_std['population_size'],
        'm': df_std['m'],
        '# of Iterations': df_std['n_iter'],
        'SD of Relative Accuracy': df_std['quality'],
        'SD of Objective Values': df_std['opt_val'],
    })

    palette = newcmp
    plt.figure(figsize=(4, 3))
    g = sns.lineplot(data=data[data['# of Iterations'] == 5],
                     x='# of Particles',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls=':')
    g = sns.lineplot(data=data[data['# of Iterations'] == 10],
                     x='# of Particles',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls='--')
    g = sns.lineplot(data=data[data['# of Iterations'] == 20],
                     x='# of Particles',
                     y='SD of Objective Values',
                     hue='m',
                     palette=palette,
                     ls='-')
    g.get_legend().remove()
    plt.tight_layout()
    plt.savefig('./result/4_opt_perf_quality_eval_std.pdf')
    plt.show()
