import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP
from sklearn.manifold import TSNE


def plot_embeddings(Y, y=None, title=None, palette='tab10'):
    if y is None:
        y = np.array([0] * Y.shape[0])

    data = {'x': Y[:, 0], 'y': Y[:, 1], 'label': y}

    plt.figure(figsize=(3, 3))
    g = sns.scatterplot(data=data,
                        x='x',
                        y='y',
                        hue='label',
                        palette=palette,
                        zorder=2)

    g.set(xticks=[])
    g.set(yticks=[])
    g.set(xlabel=None)
    g.set(ylabel=None)
    if title:
        g.set_title(title, fontsize=9)
    plt.setp(g.get_legend().get_texts(), fontsize=9)
    plt.setp(g.get_legend().get_title(), fontsize=9)
    plt.tight_layout()


if __name__ == '__main__':
    X2 = np.load('./data/two_spheres_with_three_class_attr.npy')
    X3 = np.load('./data/two_spheres_with_three_class_attr_disturbance.npy')
    y = np.load('./data/two_sphere_label.npy')

    # t-SNE examples
    Y = TSNE(n_components=2).fit_transform(X2)
    plot_embeddings(np.flip(Y), np.flip(y) + 1, 't-SNE (Dataset 2)')
    plt.savefig('./result/tsne_data2.pdf')
    plt.show()

    Y = TSNE(n_components=2).fit_transform(X3)
    plot_embeddings(np.flip(Y), np.flip(y) + 1, 't-SNE (Dataset 3)')
    plt.savefig('./result/tsne_data3.pdf')
    plt.show()

    # UMAP with different settings
    for n_neighbors in [3, 5, 10, 20, 40]:
        for min_dist in [0.0, 0.1, 0.2, 0.4, 0.8]:
            Y = UMAP(n_components=2,
                     n_neighbors=n_neighbors,
                     min_dist=min_dist).fit_transform(X2)
            plot_embeddings(
                Y, y + 1,
                f'UMAP (Data 2, k:{n_neighbors}, min_dist:{min_dist})')
            plt.savefig(
                f'./result/umap_different_hyperparams/data2_{n_neighbors}_{min_dist}.pdf'
            )
            plt.close()

            Y = UMAP(n_components=2,
                     n_neighbors=n_neighbors,
                     min_dist=min_dist).fit_transform(X3)
            plot_embeddings(Y, y + 1,
                            f'Data 3, k: {n_neighbors}, min_dist:{min_dist})')
            plt.savefig(
                f'./result/umap_different_hyperparams/data3_{n_neighbors}_{min_dist}.pdf'
            )
            plt.close()
