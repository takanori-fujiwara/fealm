import numpy as np

from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
# faster but memory leak can happen
from pathos.multiprocessing import ProcessPool as Pool
# slower but memory leak safe
# from pathos.multiprocessing import _ProcessingPool as Pool

from umap import UMAP

import fealm.graph_dissim as gd
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
    - projection_form
    - n_components (if "w" is not used as the form for feature learning)
    - n_repeats
    - pso_* (based on the problem size, you might need to set larger numebers)

    Parameters
    ----------
    n_neighbors: int, optional, (default=15)
        Number of neighbors used when constructing a graph. k in the paper.
    projection_form: string, optional (default='w')
        The projection matrix form/constraint used for optimization. Sselect
        from 'w', 'p_wMv', 'no_constraint' (or other supported options,
        'M', 'wM', 'Mv', 'wMv').
        - 'w': only feature scaling (w in the paper)
        - 'p_wMv': feature scaling and transformation (wMv in the paper).
          Note: When 'wMv', optmization is over the union of Sphere, Grasmann,
          and Sphere manifolds. But, when, 'p_wMv' (psuedo wMv) performs
          optimization over Euclidean manifold to provide more flexibility
          during the optimization; then the projection matrix will be fitted to
          the constraint of wMv.
        - 'no_constraint': apply no constraint when computing projection matrix P.
    n_components: int, optional, (default=None)
        Number of components to take if generating linear transformations
        (e.g., "wMv" in the above paper). m' in the paper. This is not needed to
        be indicated if only applying feature scaling ("w" in the above paper).
    n_repeats: int, optional, (default=5)
        Number of iterative generations of feature leraning results.
        r in the paper.
    pso_n_nonbest_solutions: int, optional, (default=10)
        Number of non-best results included. s in the paper.
    pso_nonbest_solution_selection: string, optional, (default='projection_dissim')
        Method used for the selection of non-best results. Select from
        {'projection_dissim' or 'random'}.
    pso_n_iterations: int, optional, (default=10)
        Number or optimization iterations for particle swarm optimization.
        Higher number, more optimized results.
    pso_n_jobs: int, optional, (default=-1)
        Number of processes used for particle swarm optimization. When -1, all
        available processes are used.
    pso_max_time: float, optional, (default=1000)
        Maximum time (in second) spent for particle swarm optimization.
    graph_func: function, optional, (default=None)
        Function used for graph generation. When None, k-nearest neigbor
        graph generation is used. f_Gr in the paper.
    graph_dissim: function, optional, (default=none)
        Function used for computing graph dissimilarities. d_DR in the paper.
        When None, NSD with beta=1 is used.
    graph_dissim_reduce_func: function, optional, (default=np.min)
        Reduce function used for computing overall graph
        dissimilarities. Phi in the paper.
    graph_dissim_preprocessing: bool, function, or None, optional, (default=None)
        Function returning a dictionary containing preprocessed measures that
        can be used to speed up dissimilarity computation with graph_dissim.
        When None, S1 and sig1 in NSD will be precomputed for faster computation.
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

    def __init__(self,
                 n_neighbors=15,
                 n_repeats=5,
                 projection_form='w',
                 n_components=None,
                 pso_n_nonbest_solutions=10,
                 pso_nonbest_solution_selection='projection_dissim',
                 pso_population_size=100,
                 pso_n_iterations=10,
                 pso_n_jobs=-1,
                 pso_max_time=3600,
                 graph_func=None,
                 graph_dissim=None,
                 graph_dissim_reduce_func=np.min,
                 graph_dissim_preprocessing=None):
        self.n_neighbors = n_neighbors
        self.projection_form = projection_form
        self.n_components = n_components
        self.n_repeats = n_repeats
        self.pso_n_nonbest_solutions = pso_n_nonbest_solutions
        self.pso_nonbest_solution_selection = pso_nonbest_solution_selection
        self.pso_population_size = pso_population_size
        self.pso_n_iterations = pso_n_iterations
        self.pso_n_jobs = pso_n_jobs
        self.pso_max_time = pso_max_time
        self.graph_func = graph_func
        self.graph_dissim = graph_dissim
        self.graph_dissim_reduce_func = graph_dissim_reduce_func
        self.graph_dissim_preprocessing = graph_dissim_preprocessing

        self.opt = None
        self.Ps = None
        self.best_P_indices = None

        if self.pso_n_nonbest_solutions > self.pso_population_size:
            print(
                'pso_n_nonbest_solutions must not be larger than pso_population_size. pso_n_nonbest_solutions will be set to be pso_population_size'
            )
            self.pso_n_nonbest_solutions = self.pso_population_size

        if self.graph_func is None:
            self.graph_func = lambda X: gf.nearest_nbr_graph(
                X, n_neighbors=self.n_neighbors, to_networx_graph=False)

        if self.graph_dissim is None:
            # Neighbor and Shape Dissimilarity (NSD)
            self.graph_dissim = lambda G1, G2, S1=None, sig1=None: gd.nsd(
                G1,
                G2,
                S1=S1,
                sig1=sig1,
                fixed_degree=self.n_neighbors,
                beta=1)
            self.save_snn_lsdsig = True

        if self.graph_dissim_preprocessing is None:
            self.graph_dissim_preprocessing = lambda G: {
                'S1': gd._shared_neighbor_sim(G, k=self.n_neighbors),
                'sig1': gd._lsd_trace_signature(G)
            }

    def _gen_comp_dist(self, Gs, graph_dissim):

        def comp_dist(ij):
            i, j = ij
            return graph_dissim(Gs[i], Gs[j])

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

        Gs_ = []
        if len(Gs) == 0:
            Gs_.append(self.graph_func(X))
        else:
            Gs_ += Gs

        gd_preprocessed_data = None
        if self.graph_dissim_preprocessing:
            gd_preprocessed_data = []
            for G in Gs_:
                gd_preprocessed_data.append(self.graph_dissim_preprocessing(G))

        for i in range(self.n_repeats):
            print(f'{i+1}th repeat')
            max_cost_evaluations = self.pso_population_size * self.pso_n_iterations
            solver = ParticleSwarm(max_time=self.pso_max_time,
                                   max_cost_evaluations=max_cost_evaluations,
                                   population_size=self.pso_population_size,
                                   n_jobs=self.pso_n_jobs)

            # use no_constraint when form is 'p_wMv'
            self.opt = Optimization(
                graph_func=self.graph_func,
                graph_dissim=self.graph_dissim,
                graph_dissim_reduce_func=self.graph_dissim_reduce_func,
                solver=solver,
                form=self.projection_form
                if not self.projection_form == 'p_wMv' else 'no_constraint')

            self.opt = self.opt.fit(X,
                                    Gs=Gs_,
                                    n_components=self.n_components,
                                    multiple_answers=True,
                                    gd_preprocessed_data=gd_preprocessed_data)

            tmp_Ps = self.opt.Ps

            if self.pso_n_nonbest_solutions == 0:
                new_Ps = []
            else:
                if self.pso_nonbest_solution_selection == 'projection_dissim':
                    tmp_Ps = [self._consistent_signs(P) for P in tmp_Ps]
                    if not self.projection_form == 'w':
                        tmp_Ps = [self._consistent_scales(P) for P in tmp_Ps]
                        tmp_Ps = [self._consistent_order(P) for P in tmp_Ps]

                    new_Ps = self._get_k_centers_of_Ps(
                        tmp_Ps, self.pso_n_nonbest_solutions)
                else:
                    if not self.pso_nonbest_solution_selection == 'random':
                        print(
                            'indicated pso_nonbest_solution_selection is not supported and "random" is used.'
                        )
                    indices = np.random.randint(
                        len(tmp_Ps), size=self.pso_n_nonbest_solutions)
                    new_Ps = [tmp_Ps[idx] for idx in indices]

            # append the best (duplication might happen)
            best_P = self._consistent_signs(self.opt.P)
            if not self.projection_form == 'w':
                best_P = self._consistent_scales(best_P)
                best_P = self._consistent_order(best_P)
            new_Ps.append(best_P)

            # restrict projection matrix being on wMv's manifold
            if self.projection_form == 'p_wMv':
                for i, P in enumerate(new_Ps):
                    w, M, v = self.opt.mat_decomp(P)
                    new_Ps[i] = np.diag(w) @ M @ np.diag(v)

            self.Ps += new_Ps
            self.best_P_indices.append(len(self.Ps) - 1)

            # graph based on best P
            new_G = self.graph_func(X @ new_Ps[-1])
            Gs_.append(new_G)

            if self.graph_dissim_preprocessing:
                gd_preprocessed_data.append(self.graph_dissim_preprocessing(G))

        return self

    def _embeddings_dissim_embedding(self, embedding_Ds, dr_inst=UMAP()):
        dr_inst.metric = 'precomputed'
        return dr_inst.fit_transform(embedding_Ds)

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
                               n_representatives=10,
                               Ps=None,
                               XP_dr_inst=UMAP(),
                               Ys=None,
                               Y_graph_func=None,
                               Y_graph_dissim=None,
                               Y_dr_inst=UMAP(),
                               clustering_on_emb_of_Ys=True):
        """Find representative projection matrices from Ps.
        Parameters
        ----------
        X: array-like, shape(n_samples, n_attributes)
            Target data.
        n_representatives: int, optional, (default=10)
            Number of prpjection matrices to be recommended.
        Ps: list of 2D arrays, optional, (default=None)
            List of precomputed projection matrices. If None, self.Ps will be used
        XP_dr_inst: instance of dimensionality reduction method, optional, (default=UMAP())
            This instance is used when generting Y (embedding result) from X @ P.
        Ys: list of 2D arrays, optional, (default=None)
            List of precomputed embeddings of X @ P. If None, Ys are generated
            by applying XP_dr_inst on each X @ P
        Y_graph_func: function, optional, (default=None)
            Graph generation function used to generate graphs from Ys. If None,
            self.graph_func will be used.
        Y_graph_dissim: function, optional, (default=None)
            Graph dissimilarity function used to compare Ys. If None,
            self.graph_dissim will be used.
        Y_dr_inst: instance of dimensionality reduction method, optional, (default=UMAP())
            This instance is used when generting embedding from Ys.
        clustering_on_emb_of_Ys: boolean, optional, (default=True)

        Returns
        -------
        dictionary containing various information
            - 'repr_Ps': representative projection matrices
            - 'repr_Ys': embedding results corresponding to 'repr_Ps'
            - 'repr_indices': indices of repr_Ys/repr_Ps in Ys/Ps
            - 'cluster_ids': cluster id assigned to each Y
            - 'Ys': all embeding results (corresponding to all Ps)
            - 'D_of_Ys': dissimilarities of Ys
            - 'emb_of_Ys': embedding result of Ys (embedding of embeddings)
        """
        if Ps is None:
            Ps = self.Ps

        if Ys is None:
            Ys = []
            for P in Ps:
                Ys.append(XP_dr_inst.fit_transform(X @ P))

        if Y_graph_func is None:
            Y_graph_func = self.graph_func
        if Y_graph_dissim is None:
            Y_graph_dissim = self.graph_dissim

        Gs_of_Ys = [Y_graph_func(Y) for Y in Ys]
        D_of_Ys = self._dist_comp_parallel(Gs_of_Ys, dist_func=Y_graph_dissim)

        emb_of_Ys = self._embeddings_dissim_embedding(D_of_Ys,
                                                      dr_inst=Y_dr_inst)

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
            'repr_indices': closest_Y_indices,
            'cluster_ids': cluster_ids,
            'Ys': Ys,
            'D_of_Ys': D_of_Ys,
            'emb_of_Ys': emb_of_Ys,
        }
