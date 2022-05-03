import numpy as np

from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.cluster import SpectralClustering
# faster but memory leak can happen
from pathos.multiprocessing import ProcessPool as Pool
# slower but memory leak safe
# from pathos.multiprocessing import _ProcessingPool as Pool

from umap import UMAP

import fealm.graph_dist as gd
import fealm.graph_func as gf
from fealm.solver import ParticleSwarm
from fealm.optimization import Optimization

import igraph as ig
import louvain


class FEALM():
    """FEALM. Feature learning framework for dimensionlity reduction methods.
    Implementation of the framework introduced in Fujiwara et al.,
    "Feature Learning for Dimensionality Reduction toward Maximal Extraction of
    Hidden Patterns".
    For details of parameters, please also refer to the paper.

    Parameters need to be tuned based on a dataset are:
    - n_neighbors
    - n_components (if "w" is not used as the form for feature learning)
    - n_repeats
    - pso_maxtime
    - pso_niter
    - form_and_sizes

    Parameters
    ----------
    n_neighbors: int
        Number of neighbors used when constructing a graph. k in the paper.
    n_components: int, optional, (default=None)
        Number of components to take if generating linear transformations
        (e.g., "wMv" in the above paper). m' in the paper. This is not needed to
        be indicated if only applying feature scaling ("w" in the above paper).
    graph_dist: function, optional, (default=none)
        Function used for computing graph dissimilarities. d_DR in the paper.
        When None, NSD with beta=1 is used.
    graph_dist_aggregation: function, optional, (default=np.min)
        Aggregation/reduce function used for computing overall graph
        dissimilarities. Phi in the paper.
    graph_func: function, optional, (default=None)
        Function used for graph generation. When None, k-nearest neigbor
        graph generation is used. f_Gr in the paper.
    n_repeats: int, optional, (default=5)
        Number of iterative generations of feature leraning results.
        r in the paper.
    pso_maxtime: float, optional, (default=1000)
        Maximum time (in second) spent for particle swarm optimization.
    pso_niter: int, optional, (default=10)
        Number or optimization iterations for particle swarm optimization.
        Higher number, more optimized results.
    pso_njobs: int, optional, (default=-1)
        Number of processes used for particle swarm optimization. When -1, all
        available processes are used.
    form_and_sizes: dictionary, optional (default = {'w': {'population_size': 100, 'n_results': 10, 'result_selection': 'P'}, 'p_wMv': {'population_size': 100, 'n_results': 10, 'result_selection': 'P'}})
        Dictionary indicating projection matrix forms considered in feature
        learning and the corresponding optimization settings.
        The dicitonary must contain a key indicating the matrix form but
        the optimization settings are optional.
        - form: select from 'w', 'p_wMv', 'no_constraint' (or other supported options, 'M', 'wM', 'Mv', 'wMv').
            - 'w': use only feature scaling (w in the paper)
            - 'p_wMv': use feature scaling and transformation (wMv in the paper). Note: while 'wMv' is performing optmization over Sphere and Grasmann manifolds. 'p_wMv' (psuedo wMv) is performing optimization over Euclidean manifold to provide more flexibility during the optimization. After the optimization, fit the projection matrix to the constraint of wMv.
            - 'no_constraint': apply no constraint when computing projection matrix P.
        - population_size: number of particles used for particle swarm optimization
        - n_results: number of non-best results included. s in the paper.
        - result_selection: way of selection of non-best results. 'P' or 'random'
            - 'P': based on a projection matrix similarity (recommended)
            - 'random': random sampling
    Attributes
    ----------
    Ps: list of numpy 2d array
        Projection matrices generated through feature learning
    best_P_indices: list of integers
        Indices indicating which projection matrices are corresponding to the
        best solution from the particle swarm optimization.
    ----------
    Examples
    --------
    See "sample.py" or scripts in "case_studies" directory
    """

    def __init__(
            self,
            n_neighbors,
            n_components=None,
            graph_dist=None,
            graph_dist_aggregation=np.min,
            graph_func=None,
            n_repeats=5,
            pso_maxtime=1000,
            pso_niter=10,
            pso_njobs=-1,
            form_and_sizes={
                'w': {
                    'population_size': 100,
                    'n_results': 10,
                    'result_selection': 'P'
                },
                'no_constraint': {
                    'population_size': 100,
                    'n_results': 10,
                    'result_selection': 'P'
                }
            },
            # TODO: make save_snn_lsdsig more generic
            save_snn_lsdsig=True):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.graph_dist = graph_dist
        self.graph_dist_aggregation = graph_dist_aggregation
        self.graph_func = graph_func
        self.n_repeats = n_repeats
        self.form_and_sizes = form_and_sizes
        self.save_snn_lsdsig = save_snn_lsdsig
        self.pso_maxtime = pso_maxtime
        self.pso_niter = pso_niter
        self.pso_njobs = pso_njobs

        self.opt = None
        self.Ps = None
        self.best_P_indices = None

        if self.graph_dist is None:
            # Neighbor and Shape Dissimilarity (NSD)
            self.graph_dist = lambda G1, G2, S1=None, sig1=None: gd.nsd(
                G1,
                G2,
                S1=S1,
                sig1=sig1,
                fixed_degree=self.n_neighbors,
                beta=1)
            self.save_snn_lsdsig = True

        if self.graph_func is None:
            self.graph_func = lambda X: gf.nearest_nbr_graph(
                X, n_neighbors=self.n_neighbors, to_networx_graph=False)

    def _gen_comp_dist(self, Gs, graph_dist):

        def comp_dist(ij):
            i, j = ij
            return graph_dist(Gs[i], Gs[j])

        return comp_dist

    def _frobenius(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2)**2))

    def _dist_comp_parallel(self, Xs, dist_func):
        # prepare index paris for multiprocessing
        idx_pairs = []
        for i in range(len(Xs) - 1):
            for j in range(i + 1, len(Xs)):
                idx_pairs.append((i, j))

        comp_dist = self._gen_comp_dist(Xs, dist_func)

        # multiprocessing
        with Pool() as p:
            dists = p.map(comp_dist, idx_pairs)

        D = squareform(dists)

        return D

    def _get_k_centers_of_Ps(self, Ps, k):
        D = self._dist_comp_parallel(Ps, dist_func=self._frobenius)
        kmed = KMedoids(n_clusters=k, metric='precomputed').fit(D)

        return [Ps[idx] for idx in kmed.medoid_indices_]

    def _get_k_Ps_based_on_Gs(self, Ps, k, log_scaling=True):
        Gs = [self.opt._construct_graph(X, P) for P in Ps]
        D = self._dist_comp_parallel(Gs, dist_func=self.graph_dist)

        kmed = KMedoids(n_clusters=k, metric='precomputed')
        if log_scaling:
            kmed = kmed.fit(np.log(1 + D))
        else:
            kmed = kmed.fit(D)

        return [Ps[idx] for idx in kmed.medoid_indices_]

    def _consistent_signs(self, P):
        # the blow is the same with taking a cosine sim between
        # each column and a vector with elements all one.
        # then the cosine sim is negative flip the sign
        col_sum_sign = np.sign(np.sum(P, axis=0))

        # avoid multiplying with 0 when the cosine sim = 0
        col_sum_sign[col_sum_sign == 0] = 1

        return P * col_sum_sign

    def _consistent_scales(self, P):
        # normalize each column by dividing the mean of vector lengths
        mean_length = np.mean(np.linalg.norm(P, axis=0))

        return P / mean_length

    def _consistent_order(self, P):
        # TODO: need to consider a better order of axes

        # This orders columns based on their cosine similarity to a vector
        # with elements all one.
        col_sum_sign = np.sign(np.sum(P, axis=0))
        idx_sort_by_col_sum_sign = np.argsort(col_sum_sign)

        return P[:, idx_sort_by_col_sum_sign]

    def fit(self, X, Gs=[]):
        """Perform feature learning on input data.

        Parameters
        ----------
        X: array-like, shape(n_samples, n_attributes)
            Target data.
        Gs: list of graphs, optional, (default=[])
            Already produced graphs that will be referred when producing a new
            graph (if existed). A set of graphs, Gi, in the paper.
        Returns
        -------
        self.
        """
        self.Ps = []
        self.best_P_indices = []

        for form in self.form_and_sizes:
            Gs_ = []
            if len(Gs) == 0:
                Gs_.append(self.graph_func(X))
            else:
                Gs_ += Gs

            gd_prerpcessed_data = None
            if self.save_snn_lsdsig:
                gd_prerpcessed_data = []
                for G in Gs_:
                    S = gd._shared_neighbor_sim(G, k=self.n_neighbors)
                    sig = gd._lsd_trace_signature(G)
                    gd_prerpcessed_data.append({'S1': S, 'sig1': sig})

            sizes = self.form_and_sizes[form]
            for i in range(self.n_repeats):
                print(f'{i+1}th repeat')

                population_size = 100 if not 'population_size' in sizes else sizes[
                    'population_size']
                n_results = max(1, int(
                    polulationsize /
                    10)) if not 'n_results' in sizes else sizes['n_results']
                result_selection = 'P' if not 'result_selection' in sizes else sizes[
                    'result_selection']

                max_cost_evaluations = population_size * self.pso_niter

                solver = ParticleSwarm(
                    max_time=self.pso_maxtime,
                    max_cost_evaluations=max_cost_evaluations,
                    population_size=population_size,
                    n_jobs=self.pso_njobs)

                # use no_constraint when form is 'p_wMv' to avoid computations
                # for 'mat_decomp'
                form_ = form
                if form == 'p_wMv':
                    form_ = 'no_constraint'

                self.opt = Optimization(
                    graph_func=self.graph_func,
                    graph_dist=self.graph_dist,
                    graph_dist_aggregation=self.graph_dist_aggregation,
                    solver=solver,
                    form=form_)

                self.opt = self.opt.fit(
                    X,
                    Gs=Gs_,
                    n_components=self.n_components,
                    multiple_answers=True,
                    gd_prerpcessed_data=gd_prerpcessed_data)

                tmp_Ps = self.opt.Ps

                if result_selection == 'random':
                    indices = np.random.randint(len(tmp_Ps), size=n_results)
                    new_Ps = [tmp_Ps[idx] for idx in indices]
                elif result_selection == 'P':
                    # make consistent signs to avoid high dissim when sign flipped
                    # TODO: we should solve arbitrary column order as well
                    tmp_Ps = [self._consistent_signs(P) for P in tmp_Ps]
                    if not form == 'w':
                        tmp_Ps = [self._consistent_scales(P) for P in tmp_Ps]
                        tmp_Ps = [self._consistent_order(P) for P in tmp_Ps]

                    new_Ps = self._get_k_centers_of_Ps(tmp_Ps, n_results)

                elif result_selection == 'G':
                    new_Ps = self._get_k_Ps_based_on_Gs(tmp_Ps, n_results)
                elif result_selection == 'G_without_log':
                    new_Ps = self._get_k_Ps_based_on_Gs(tmp_Ps,
                                                        n_results,
                                                        log_scaling=False)

                # append the best (duplication might happen)
                best_P = self._consistent_signs(self.opt.P)
                new_Ps.append(best_P)

                if form == 'p_wMv':
                    for i, P in enumerate(new_Ps):
                        w, M, v = self.opt.mat_decomp(P)
                        new_Ps[i] = np.diag(w) @ M @ np.diag(v)

                self.Ps += new_Ps
                self.best_P_indices.append(len(self.Ps) - 1)

                # graph based on best P
                new_G = self.graph_func(X @ new_Ps[-1])
                Gs_.append(new_G)

                if self.save_snn_lsdsig:
                    S = gd._shared_neighbor_sim(new_G, k=self.n_neighbors)
                    sig = gd._lsd_trace_signature(new_G)

                    gd_prerpcessed_data.append({'S1': S, 'sig1': sig})

        return self

    def _embeddings_dissim_embedding(self,
                                     embedding_Ds,
                                     dr_class=UMAP,
                                     dr_kwargs={
                                         'n_components': 2,
                                         'n_neighbors': 15,
                                         'min_dist': 0.1
                                     }):
        return dr_class(metric='precomputed',
                        **dr_kwargs).fit_transform(np.log(1 + embedding_Ds))

    def _clustering_by_emb_of_Ys(self, emb_of_Ys, n_representatives):
        cluster_ids = SpectralClustering(
            n_clusters=n_representatives,
            assign_labels='discretize').fit(emb_of_Ys).labels_

        # reordering cluster IDs based on x posisiton of emb_of_Ys
        # to avoid randomly changing IDs when using together with UI
        mean_xpos = []
        for label in np.unique(cluster_ids):
            related_indices = np.where(cluster_ids == label)[0]
            mean_xpos.append(
                np.array(emb_of_Ys)[related_indices, :].mean(axis=0)[0])
        xpos_rank = rankdata(mean_xpos, method='ordinal') - 1

        cluster_ids = np.array([xpos_rank[c_id] for c_id in cluster_ids])

        return cluster_ids

    def _clustering_by_D_of_Ys(self, D_of_Ys, n_representatives):
        cluster_ids = SpectralClustering(
            n_clusters=n_representatives,
            assign_labels='discretize',
            affinity='precomputed').fit(D_of_Ys).labels_

        return cluster_ids

    def find_representative_Ps(self,
                               X,
                               X2Y_dr_inst,
                               Ps=None,
                               Ys=None,
                               n_representatives=10,
                               include_X=False,
                               graph_dist=None,
                               graph_func=None,
                               X2Y_n_neighbor_scaling={
                                   'snn': 1,
                                   'netlsd': 1
                               },
                               Ys_dr_class=UMAP,
                               Ys_dr_kwargs={
                                   'n_components': 2,
                                   'n_neighbors': 15,
                                   'min_dist': 0.1
                               },
                               clustering_on_emb_of_Ys=False):
        """Find representative projection matrices from Ps.
        Note: here only important parameters are described.
        Parameters
        ----------
        X: array-like, shape(n_samples, n_attributes)
            Target data.
        X2Y_dr_inst: instance of dimensionality reduction method
            This instance is used when generting Y (embedding result) from X.
        n_representatives: int, optional, (default=10)
            Number of prpjection matrices to be recommended.
        Returns
        -------
        dictionary containing various information
            - 'repr_Ps': representative projection matrices
            - 'repr_Ys': embedding results corresponding to 'repr_Ps'
            - 'closest_Y_indices': indices of closest Y to the cluster centers
            - 'Ys': all embeding results (corresponding to all Ps)
            - 'D_of_Ys': dissimilarities of Ys
            - 'emb_of_Ys': embedding result of Ys (embedding of embeddings)
            - 'cluster_ids': cluster id assigned to each Y
        """
        if Ps is None:
            Ps = self.Ps

        if Ys is None:
            Ys = []
            for P in Ps:
                Ys.append(X2Y_dr_inst.fit_transform(X @ P))

        if graph_dist is None:
            graph_dist = self.graph_dist
        if graph_func is None:
            graph_func = self.graph_func

        Gs_of_Ys = [graph_func(Y) for Y in Ys]
        D_of_Ys = self._dist_comp_parallel(Gs_of_Ys, dist_func=graph_dist)

        emb_of_Ys = self._embeddings_dissim_embedding(D_of_Ys,
                                                      dr_class=Ys_dr_class,
                                                      dr_kwargs=Ys_dr_kwargs)

        if clustering_on_emb_of_Ys:
            cluster_ids = self._clustering_by_emb_of_Ys(
                emb_of_Ys, n_representatives)
        else:
            cluster_ids = self._clustering_by_D_of_Ys(D_of_Ys,
                                                      n_representatives)

        # get Y closeset to each cluster center
        closest_Y_indices = []
        for label in np.unique(cluster_ids):
            related_indices = np.where(cluster_ids == label)[0]
            mean_pos = emb_of_Ys[related_indices, :].mean(axis=0)

            closest_Y_idx = related_indices[np.argsort(
                np.sum((emb_of_Ys[related_indices, :] - mean_pos)**2,
                       axis=1))[0]]
            closest_Y_indices.append(closest_Y_idx)

        repr_Ys = [Ys[idx] for idx in closest_Y_indices]
        repr_Ps = [Ps[idx] for idx in closest_Y_indices]

        return {
            'repr_Ps': repr_Ps,
            'repr_Ys': repr_Ys,
            'closest_Y_indices': closest_Y_indices,
            'Ys': Ys,
            'D_of_Ys': D_of_Ys,
            'emb_of_Ys': emb_of_Ys,
            'cluster_ids': cluster_ids
        }


# # potential way to cluster embedding results instead of SpectralClustering
# # this clustering doesn't require specifying k
# def cluster_by_modularity(
#         data,
#         n_neighbors=15,
#         graph_type='simple',
#         partition_type=louvain.RBConfigurationVertexPartition,
#         resolution_parameter=2):
#     '''
#     graph_type: 'simple', 'fuzzy', 'precomputed'
#     partition_type: louvain.RBConfigurationVertexPartition, etc.
#     '''
#
#     if graph_type == 'precomputed':
#         A = data
#     elif graph_type == 'simple':
#         A = gf.nearest_nbr_graph(data,
#                                  n_neighbors=n_neighbors,
#                                  to_networx_graph=False)
#     elif graph_type == 'fuzzy':
#         A = gf.fuzzy_nearest_nbr_graph(data,
#                                        n_neighbors=n_neighbors,
#                                        to_networx_graph=False)
#     sources, targets = A.nonzero()
#     edges = zip(sources.tolist(), targets.tolist())
#     weights = A[sources, targets].A1
#     g = ig.Graph(directed=False)
#     g.add_vertices(A.shape[0])
#     edges = list(zip(sources, targets))
#     g.add_edges(edges)
#     g.es['weight'] = weights
#
#     partition = louvain.find_partition(
#         g,
#         partition_type=partition_type,
#         weights=np.array(g.es['weight']),
#         resolution_parameter=resolution_parameter)
#
#     return np.array(partition.membership)
