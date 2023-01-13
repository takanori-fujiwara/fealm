import numpy as np
from sklearn.preprocessing import scale


def make_spheres(aiming_n_samples=150, factor=0.4, noise=0.06, random_state=0):
    from sklearn.utils import check_random_state

    if factor >= 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")

    n_samples_in = aiming_n_samples // 3
    n_samples_out = aiming_n_samples - n_samples_in

    n_samples_out1 = int(np.sqrt(n_samples_out / 2))
    n_samples_out2 = 2 * n_samples_out1
    n_samples_in1 = int(np.sqrt(n_samples_in / 2))
    n_samples_in2 = 2 * n_samples_in1

    generator = check_random_state(random_state)
    linspace_phi_out = np.linspace(0, np.pi, n_samples_out1, endpoint=False)
    linspace_theta_out = np.linspace(0,
                                     2 * np.pi,
                                     n_samples_out2,
                                     endpoint=False)
    linspace_phi_in = np.linspace(0, np.pi, n_samples_in1, endpoint=False)
    linspace_theta_in = np.linspace(0,
                                    2 * np.pi,
                                    n_samples_in2,
                                    endpoint=False)

    outer_circ_x = np.outer(np.sin(linspace_theta_out),
                            np.cos(linspace_phi_out))
    outer_circ_y = np.outer(np.sin(linspace_theta_out),
                            np.sin(linspace_phi_out))
    outer_circ_z = np.outer(np.cos(linspace_theta_out),
                            np.ones_like(linspace_phi_out))
    inner_circ_x = np.outer(np.sin(linspace_theta_in),
                            np.cos(linspace_phi_in)) * factor
    inner_circ_y = np.outer(np.sin(linspace_theta_in),
                            np.sin(linspace_phi_in)) * factor
    inner_circ_z = np.outer(np.cos(linspace_theta_in),
                            np.ones_like(linspace_phi_in)) * factor

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y),
        np.append(outer_circ_z, inner_circ_z)
    ]).T
    y = np.hstack([
        np.zeros(n_samples_out1 * n_samples_out2, dtype=np.intp),
        np.ones(n_samples_in1 * n_samples_in2, dtype=np.intp)
    ])

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


if __name__ == '__main__':
    visualize_data = True

    ### Dataset 1
    aiming_n_samples = 300
    two_spheres, y = make_spheres(aiming_n_samples=aiming_n_samples,
                                  factor=0.4,
                                  noise=0.06,
                                  random_state=0)
    n_samples = two_spheres.shape[0]

    ### Dataset 2
    n_adding_attrs = 1
    n_subsamples = n_samples // 3
    adding_attrs = np.random.normal(loc=10,
                                    scale=0.1,
                                    size=(n_samples, n_adding_attrs))
    adding_attrs[:n_subsamples] = np.random.normal(loc=0,
                                                   scale=0.1,
                                                   size=(n_subsamples,
                                                         n_adding_attrs))
    adding_attrs[n_subsamples:2 * n_subsamples] = np.random.normal(
        loc=4, scale=0.1, size=(n_subsamples, n_adding_attrs))
    y2 = np.zeros(n_samples)
    y2[n_subsamples:2 * n_subsamples] = 1
    y2[2 * n_subsamples:3 * n_subsamples] = 2

    # shuffling the order of samples
    shuffled_indices = np.array(range(n_samples))
    np.random.shuffle(shuffled_indices)
    adding_attrs = adding_attrs[shuffled_indices, :]
    y2 = y2[shuffled_indices]
    y2 = y2.astype(int)

    two_spheres_with_additional_attrs = np.hstack((two_spheres, adding_attrs))

    ### Dataset3
    ratio = 0.2
    two_spheres_with_add_attrs_distrub = np.zeros((n_samples, 4))
    disturb = scale(adding_attrs[:, 0])
    two_spheres_with_add_attrs_distrub[:, 0] = two_spheres[:, 0]
    two_spheres_with_add_attrs_distrub[:, 1] = two_spheres[:, 1]
    two_spheres_with_add_attrs_distrub[:,
                                       2] = two_spheres[:, 2] + ratio * disturb
    two_spheres_with_add_attrs_distrub[:, 3] = disturb

    X1 = scale(two_spheres)
    X2 = scale(two_spheres_with_additional_attrs)
    X3 = scale(two_spheres_with_add_attrs_distrub)

    np.save('./data/two_spheres.npy', X1)
    np.save('./data/two_spheres_with_three_class_attr.npy', X2)
    np.save('./data/two_spheres_with_three_class_attr_disturbance.npy', X3)
    np.save('./data/two_sphere_label.npy', y)
    np.save('./data/three_class_label.npy', y2)

    ### Visualization (if needed)
    if visualize_data:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        from umap import UMAP
        from sklearn.decomposition import PCA

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        colors = ['tab:blue', 'tab:orange']
        for label in y:
            ax.scatter(two_spheres[y == label, 0],
                       two_spheres[y == label, 1],
                       two_spheres[y == label, 2],
                       alpha=0.8,
                       s=20,
                       c=colors[label])
        ax.set_box_aspect((np.ptp(two_spheres[:, 0]), np.ptp(two_spheres[:,
                                                                         1]),
                           np.ptp(two_spheres[:, 2])))
        plt.savefig('./result/two_spheres.pdf')
        plt.show()

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

        Y = PCA(n_components=2).fit_transform(X1)
        plot_embeddings(np.flip(Y), np.flip(y) + 1, 'PCA (Dataset 1)')
        plt.savefig('./result/pca_data1.pdf')
        plt.show()

        Y = UMAP(n_components=2, n_neighbors=15,
                 min_dist=0.1).fit_transform(X1)
        plot_embeddings(Y, y + 1, 'UMAP (Dataset 1)')
        plt.savefig('./result/umap_data1.pdf')
        plt.show()

        Y = UMAP(n_components=2, n_neighbors=15,
                 min_dist=0.1).fit_transform(X2)
        plot_embeddings(Y, y + 1, 'UMAP (Dataset 2)')
        plt.savefig('./result/umap_data2.pdf')
        plt.show()
        plot_embeddings(Y, y2 + 1, 'UMAP (Dataset 2)', palette='Dark2_r')
        plt.savefig('./result/umap_data2_2.pdf')
        plt.show()

        Y = UMAP(n_components=2, n_neighbors=15,
                 min_dist=0.1).fit_transform(X3)
        plot_embeddings(Y, y + 1, 'UMAP (Dataset 3)')
        plt.savefig('./result/umap_data3.pdf')
        plt.show()
        plot_embeddings(Y, y2 + 1, 'UMAP (Dataset 3)', palette='Dark2_r')
        plt.savefig('./result/umap_data3_2.pdf')
        plt.show()

        Y = UMAP(n_components=2, n_neighbors=15,
                 min_dist=0.1).fit_transform(X3[:, :3])
        plot_embeddings(Y, y + 1, 'UMAP (Dataset 3, first 3 attrs)')
        plt.savefig('./result/umap_data3_sub.pdf')
        plt.show()
        plot_embeddings(Y,
                        y2 + 1,
                        'UMAP (Dataset 3, irst 3 attrs)',
                        palette='Dark2_r')
        plt.savefig('./result/umap_data3_sub_2.pdf')
        plt.show()
