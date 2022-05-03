import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import preprocessing

from umap import UMAP

from fealm.opt_set_proj import OptSetProj
import fealm.plot as fplot

import pyreadstat  # please install via pip (pip3 install pyreadstat)
from ppic_meta import metainfo

if __name__ == '__main__':
    tableau10 = {
        'teal': '#78B7B2',
        'blue': '#507AA6',
        'orange': '#F08E39',
        'red': '#DF585C',
        'green': '#5BA053',
        'purple': '#AF7BA1',
        'yellow': '#ECC854',
        'brown': '#9A7460',
        'pink': '#FD9EA9',
        'gray': '#BAB0AC'
    }

    n_results = 20
    opt = OptSetProj()

    #########
    # 1. Wine
    #########
    dataset = datasets.load_wine()
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    feat_names = dataset.feature_names
    inst_names = np.arange(0, X.shape[1])
    target_colors = np.array(
        [tableau10['blue'], tableau10['orange'], tableau10['green']])

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    n_neighbors = 15
    min_dist = 0.1

    # select only od280/od315
    X_od = X[:, feat_names.index('od280/od315_of_diluted_wines')][:,
                                                                  np.newaxis]
    Y = UMAP(n_components=2, n_neighbors=n_neighbors,
             min_dist=min_dist).fit_transform(X_od)
    fig = fplot.plot_embeddings([Y],
                                y + 1,
                                palette=sns.color_palette([
                                    tableau10['blue'], tableau10['orange'],
                                    tableau10['green']
                                ]))
    fig.axes[0].set_title('Using only od280/od315')
    plt.savefig(f'./result/supplementary/od280od315_wine.pdf')
    plt.show()

    # select only color intensity
    X_ci = X[:, feat_names.index('color_intensity')][:, np.newaxis]
    Y = UMAP(n_components=2, n_neighbors=n_neighbors,
             min_dist=min_dist).fit_transform(X_ci)
    fig = fplot.plot_embeddings([Y],
                                y + 1,
                                palette=sns.color_palette([
                                    tableau10['blue'], tableau10['orange'],
                                    tableau10['green']
                                ]))
    fig.axes[0].set_title('Using only color intensity')
    plt.savefig(f'./result/supplementary/colorintensity_wine.pdf')
    plt.show()

    #########
    # 2. PPIC
    #########
    df, meta = pyreadstat.read_sav('./data/ppic/2018.10.24.release.sav')

    # covert code to binary, categorical, numeric numbers
    for col_key in df:
        df[col_key] = np.vectorize(metainfo[col_key]['code_num'])(df[col_key])

    # remove attributes that we are not interested in
    for col_key in df:
        if metainfo[col_key]['use'] == 0:
            del df[col_key]

    # we only want to compare democrat vs republican supporters
    # 1: Democrat, 2: Republican
    df.drop(np.where(df['q4a'] > 2)[0], axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # combine planned- and used-vote method
    df['q37a'][df['q37a'] == 999] = df['q37b'][df['q37a'] == 999]
    df.drop(['q37b'], axis=1, inplace=True)

    # also drop other non-important attributes to reduce # of nan
    # (e.g., there is a similar quesiton or related to the dropped cols)
    # 'q10: satisfy_senate_choice',
    # 'q12: congress_trump',
    # 'q15: importance_proposition6',
    # 'q17: imortance_proposition10',
    df.drop(['q10', 'q12', 'q15', 'q17'], axis=1, inplace=True)

    # remove attributes that have many nan values
    col_na_ratios = np.array(
        [np.sum(df[col_key] == 999) / df.shape[0] for col_key in df])
    df.drop(df.columns[np.where(col_na_ratios > 0.1)[0]], axis=1, inplace=True)

    # drop subjects who have NA in their answers
    row_na_ratios = np.array(
        [np.sum(row == 999) / df.shape[1] for index, row in df.iterrows()])
    df = df.iloc[row_na_ratios <= 0.0, :]
    df.reset_index(inplace=True, drop=True)

    # use q4a as label
    y = np.array(df['q4a'])
    y[y == 1] = 0  # democrat
    y[y == 2] = 1  # republican
    y_to_name = {0: 'Dem', 1: 'Rep'}
    target_names = np.array(['Dem', 'Rep'])
    target_colors = np.array([tableau10['blue'], tableau10['red']])

    df.drop(['q4a'], axis=1, inplace=True)
    X = np.array(df)
    inst_names = df.index

    feat_names = []
    name_to_q = {}
    for col_key in df.columns:
        name = col_key
        feat_names.append(name)
        name_to_q[name] = col_key
    feat_names = np.array(feat_names)

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # select only Q24 and Q31
    X_q24q31 = X[:, (feat_names == 'q24') | (feat_names == 'q31')]
    Y = UMAP(n_components=2, n_neighbors=n_neighbors,
             min_dist=min_dist).fit_transform(X_q24q31)
    fig = fplot.plot_embeddings([Y],
                                y + 1,
                                palette=sns.color_palette(
                                    [tableau10['blue'], tableau10['red']]))
    fig.axes[0].set_title('Using only Q24 and Q31')
    plt.savefig(f'./result/supplementary/q24q31_ppic.pdf')
    plt.show()

    # select only Q18, Q19, and D7
    X_q18q19d7 = X[:, (feat_names == 'q18') | (feat_names == 'q19') |
                   (feat_names == 'd7')]
    Y = UMAP(n_components=2, n_neighbors=n_neighbors,
             min_dist=min_dist).fit_transform(X_q18q19d7)
    fig = fplot.plot_embeddings([Y],
                                y + 1,
                                palette=sns.color_palette(
                                    [tableau10['blue'], tableau10['red']]))
    fig.axes[0].set_title('Using only Q18, Q19, and D7')
    plt.savefig(f'./result/supplementary/q18q19d7_ppic.pdf')
    plt.show()
