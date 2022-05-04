import time
import numpy as np
import pandas as pd

import fealm.graph_func as gf
import fealm.graph_dissim as gd
from fealm.fealm import FEALM

if __name__ == '__main__':
    k = 15
    m = 10
    m_prime = int(m / 2)
    ns = [50, 100, 200, 400, 800, 1600]
    list_of_n_jobs = [1, 2, 4, 8, 16]
    to_data_name = lambda n: f'./data/document_vec_n{n}_m{m}.npy'

    result = []
    for n in ns:
        X = np.load(to_data_name(n))
        for n_jobs in list_of_n_jobs:
            fealm = FEALM(n_neighbors=k,
                          projection_form='no_constraint',
                          n_components=m_prime,
                          n_repeats=1,
                          pso_population_size=500,
                          pso_n_nonbest_solutions=0,
                          pso_n_iterations=10,
                          pso_n_jobs=n_jobs)
            s = time.time()
            fealm = fealm.fit(X)
            e = time.time()

            result.append({'n': n, 'n_jobs': n_jobs, 'time': e - s})
            print(f'n{n} {n_jobs} {e - s}')

    pd.DataFrame(result).to_csv('./result/3_fealm_perf_eval.csv', index=False)

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    df = pd.read_csv('./result/3_fealm_perf_eval.csv')
    df['speedup'] = df['time']

    for n in ns:
        base = df[(df['n'] == n) & (df['n_jobs'] == 1)]['time']
        df.loc[df['n'] == n, df.columns ==
               'speedup'] = float(base) / df['speedup'][df['n'] == n]

    data = pd.DataFrame({
        'Completion Time (sec)': df['time'][df['n_jobs'] == 8],
        'n': df['n'][df['n_jobs'] == 8]
    })

    plt.figure(figsize=(4, 2.5))
    g = sns.lineplot(data=data, x='n', y='Completion Time (sec)', color='gray')
    plt.xlabel(r'$n$')
    plt.ylim(0, data['Completion Time (sec)'].max() * 1.05)
    plt.tight_layout()
    plt.savefig('./result/3_fealm_perf_eval_time.pdf')
    plt.show()

    data = pd.DataFrame({
        '# of Processors': df['n_jobs'],
        'Speedup': df['speedup'],
        'Completion Time': df['time'],
        'n': df['n'],
        'log_n': np.log(df['n'])
    })

    plt.figure(figsize=(4, 2.5))
    g = sns.lineplot(data=data,
                     x='# of Processors',
                     y='Speedup',
                     hue='log_n',
                     palette='flare')
    g.xaxis.set_major_locator(MaxNLocator(integer=True))
    g.legend(
        ns,
        title=r'$n$',
    )
    g.get_legend().remove()
    plt.xlim(0.5, 18)
    plt.ylim(0, data['Speedup'].max() * 1.05)
    plt.tight_layout()
    plt.savefig('./result/3_fealm_perf_eval_scaling.pdf')
    plt.show()
