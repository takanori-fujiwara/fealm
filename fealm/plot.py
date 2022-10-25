import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import LineCollection


def plot_embeddings(Ys, y, palette='tab10', start_title_id=0, linewidth=0):
    figsize_x = len(Ys) * 3.2
    fig, axs = plt.subplots(ncols=len(Ys), figsize=(figsize_x, 3.5))
    for i, Y in enumerate(Ys):
        data = {'x': Y[:, 0], 'y': Y[:, 1], 'label': y}
        ax = axs[i] if isinstance(axs, np.ndarray) else axs
        sns.scatterplot(data=data,
                        x='x',
                        y='y',
                        hue='label',
                        ax=ax,
                        palette=palette,
                        linewidth=linewidth,
                        zorder=2)

        ax.set(xticks=[])
        ax.set(yticks=[])
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.set_title(f'Embeddding {i + start_title_id}', fontsize=9)
        plt.setp(ax.get_legend().get_texts(), fontsize=9)
        plt.setp(ax.get_legend().get_title(), fontsize=9)
    plt.tight_layout()

    return fig


# add edges as line collection in fig
def add_edges(ax,
              vertex_positions,
              edges,
              color='#BAB0AC',
              alpha=0.5,
              linewidth=1,
              vertex_filter=None):
    edge_pos = []
    related_vertices = np.copy(vertex_filter)
    if vertex_filter is None:
        edge_pos = np.asarray([(vertex_positions[e[0]], vertex_positions[e[1]])
                               for e in edges])
    else:
        for e in edges:
            related_vertices[e[0]] = vertex_filter[e[0]]
            related_vertices[e[1]] = vertex_filter[e[1]]
            if vertex_filter[e[0]] and vertex_filter[e[1]]:
                edge_pos.append(
                    (vertex_positions[e[0]], vertex_positions[e[1]]))

    edge_collection = LineCollection(edge_pos,
                                     colors=color,
                                     linewidths=linewidth,
                                     antialiaseds=(1, ),
                                     alpha=alpha)
    edge_collection.set_zorder(1)
    ax.add_collection(edge_collection)

    return related_vertices
