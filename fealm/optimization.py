# https://github.com/pymanopt/pymanopt/blob/master/examples/low_rank_matrix_approximation.py
# https://pymanopt.org/docs/latest/manifolds.html#pymanopt.manifolds.manifold.Manifold

import autograd.numpy as np

import pymanopt
from pymanopt.manifolds.product import Product
from pymanopt.manifolds import Sphere
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.manifolds import Grassmann
from pymanopt.manifolds.oblique import Oblique
from pymanopt.manifolds.euclidean import Euclidean

from pymanopt.optimizers.steepest_descent import SteepestDescent
from fealm.optimizer import AdaptiveNelderMead


class ModifiedEuclidean(Euclidean):
    # Note: the original pymanopt implementation has a bug for
    # random_point, zero_vector
    def random_point(self):
        return np.random.normal(size=self._shape[0])

    def zero_vector(self, point):
        return np.zeros(self._shape[0])


class ModifiedOblique(Oblique):
    # Note: the original pymanopt implementation does not have _shape
    # def __init__(self, m: int, n: int):
    #     self._shape = (m, n)
    #     super().__init__(m, n)

    # # original normalize column can make NaN (when all elements are 0)
    # def _normalize_columns(self, array):
    #     col_norm = np.linalg.norm(array, axis=0)[np.newaxis, :]
    #     return np.where(col_norm == 0, 1 / array.shape[0]**0.5,
    #                     array / col_norm)
    #
    # # same with the above
    # def random_tangent_vector(self, point):
    #     vector = np.random.normal(size=point.shape)
    #     tangent_vector = self.projection(point, vector)
    #     tangent_vector_norm = self.norm(point, tangent_vector)
    #
    #     return np.where(tangent_vector_norm == 0, 1 / len(tangent_vector)**0.5,
    #                     tangent_vector / tangent_vector_norm)

    def log(self, point_a, point_b):
        vector = self.projection(point_a, point_b - point_a)
        distances = np.arccos((point_a * point_b).sum(0))
        norms = np.sqrt((vector**2).sum(0)).real
        # Try to avoid zero-division when both distances and norms are almost
        # zero.
        epsilon = np.finfo(np.float64).eps

        # actually when distances are really zero, np.arccos returns nan
        distances[np.isnan(distances)] = np.finfo(np.float64).eps

        factors = (distances + epsilon) / (norms + epsilon)

        return vector * factors


class Optimization():
    '''
    Args:
    ----------
    graph_func: graph generation function
    graph_dissim: graph dissimilarity calculation function
    graph_dissim_reduce_func: graph dissimilarity reduce function, optional (default: np.min)
    optimizer: optimizer to use, optional (default: AdaptiveNelderMead())
    form: {'w', 'M', 'wM', 'Mv', 'wMv', 'p_wMv', 'no_constraint'}
        The projection matrix form/constraint used for optimization. Sselect
        Note: the pseudo Karcher mean of Grasmann is not implemented yet
        and cannot be used (Adaptive)NelderMead for constraints involving
        the Grasmann manifold (i.e., M, wM, Mv, wMv). Use 'p_wMv' (pseudo wMv), instead.
        (https://www.cis.upenn.edu/~cis5150/Diffgeom-Grassmann.Absil.pdf).
        - 'w': only feature scaling (w in the paper)
        - 'p_wMv': feature scaling and transformation (wMv in the paper).
          Note: When 'wMv', optmization is over the union of Sphere, Grasmann,
          and Sphere manifolds. But, when, 'p_wMv' (psuedo wMv) performs
          optimization over Oblique manifold to provide more flexibility
          during the optimization; then the projection matrix will be fitted to
          the constraint of wMv.
        - 'no_constraint': apply no constraint when computing projection matrix P.
    n_trials_for_mat_decomp: int, optional, (default: 5)
        Only used when form='p_wMv'. When 'p_wMv', decompose an obtained
        projection matrix into the form of w, M, v.
    '''

    def __init__(self,
                 graph_func,
                 graph_dissim,
                 graph_dissim_reduce_func=np.min,
                 optimizer=AdaptiveNelderMead(),
                 form='no_constraint',
                 n_trials_for_mat_decomp=5):
        self.graph_func = graph_func
        self.graph_dissim = graph_dissim
        self.graph_dissim_reduce_func = graph_dissim_reduce_func
        self.optimizer = optimizer
        self.form = form
        self.n_trials_for_mat_decomp = n_trials_for_mat_decomp

        self.problem = None

        self.w = None
        self.v = None
        self.M = None
        self.P = None

        self.ws = None
        self.vs = None
        self.Ms = None
        self.Ps = None

    def _construct_graph(self, X, W):
        G = self.graph_func(X @ W)

        return G

    def _eval_dissim_graph(self, new_G, Gs, gd_params, gd_preprocessed_data):
        # gd_preprocessed_data is like: [{'S1': S1, 'D1': D1}]
        if gd_preprocessed_data is None:
            dissims = [
                self.graph_dissim(prev_G, new_G, **gd_params) for prev_G in Gs
            ]
        else:
            dissims = [
                self.graph_dissim(prev_G, new_G, **{
                    **gd_params,
                    **gd_params2
                }) for prev_G, gd_params2 in zip(Gs, gd_preprocessed_data)
            ]

        val = self.graph_dissim_reduce_func(dissims)

        return val

    def _regularization_penalty(self, P, lasso_coeff=0, ridge_coeff=0):
        l1_sum = 0
        l2_sum = 0
        if lasso_coeff != 0:
            l1_sum = np.abs(P).sum(axis=0).sum()
        # here l2 reg is applied to compare each column of P
        if ridge_coeff != 0:
            l2_sum = np.sqrt((P**2).sum(axis=1)).sum()

        return lasso_coeff * l1_sum + ridge_coeff * l2_sum

    def _eval_cost(self,
                   X,
                   P,
                   Gs,
                   lasso_coeff=0,
                   ridge_coeff=0,
                   gd_params={},
                   gd_preprocessed_data=None):
        new_G = self._construct_graph(X, P)

        dissim = self._eval_dissim_graph(
            new_G,
            Gs,
            gd_params=gd_params,
            gd_preprocessed_data=gd_preprocessed_data)
        regularization_penalty = self._regularization_penalty(
            P, lasso_coeff=lasso_coeff, ridge_coeff=ridge_coeff)

        dissim_with_penalty = dissim - regularization_penalty
        # print(dissim, regularization_penalty, dissim_with_penalty)

        # cost = 1 / dissim_with_penalty if dissim_with_penalty > 0 else np.inf
        cost = -dissim_with_penalty

        return cost

    def _gen_cost_func(self,
                       manifold,
                       form,
                       X,
                       Gs,
                       lasso_coeff=0,
                       ridge_coeff=0,
                       gd_params={},
                       gd_preprocessed_data=None):
        '''
        form: w, M, wM, Mv, wMv, p_wMv or no_constraint
        '''
        vec_len = np.sqrt(X.shape[1])

        comp_P = None
        if form == 'w':
            # np.diag(w)
            @pymanopt.function.autograd(manifold)
            def _cost_func(u):
                return self._eval_cost(
                    X,
                    P=np.diag(vec_len * u),
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)
        elif form == 'M':
            # M
            @pymanopt.function.autograd(manifold)
            def _cost_func(M):
                return self._eval_cost(
                    X,
                    P=M,
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)
        elif form == 'wM':
            # np.diag(w) @ M
            @pymanopt.function.autograd(manifold)
            def _cost_func(u, M):
                return self._eval_cost(
                    X,
                    P=np.diag(vec_len * u) @ M,
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)
        elif form == 'Mv':
            # M @ np.diag(v)
            @pymanopt.function.autograd(manifold)
            def _cost_func(M, u):
                return self._eval_cost(
                    X,
                    P=M @ np.diag(np.sqrt(len(u)) * u),
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)
        elif form == 'wMv':
            # np.diag(w) @ M @ np.diag(v)
            @pymanopt.function.autograd(manifold)
            def _cost_func(u1, M, u2):
                return self._eval_cost(
                    X,
                    P=np.diag(vec_len * u1) @ M @ np.diag(
                        np.sqrt(len(u2)) * u2),
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)
        elif form == 'no_constraint':

            @pymanopt.function.autograd(manifold)
            def _cost_func(P):
                return self._eval_cost(
                    X,
                    P=P,
                    Gs=Gs,
                    lasso_coeff=lasso_coeff,
                    ridge_coeff=ridge_coeff,
                    gd_params=gd_params,
                    gd_preprocessed_data=gd_preprocessed_data)

        # TODO: using *args can make here simpler, but it doesn't work with autograd
        # @pymanopt.function.Autograd
        # def _cost_func(*args):
        #     return self._eval_cost(comp_P(args),
        #                            Gs,
        #                            lasso_coeff=lasso_coeff,
        #                            ridge_coeff=ridge_coeff)

        return _cost_func

    def _gen_manifold(self, form, n_attrs, n_components=None):
        '''
        form: w, M, wM, Mv, wMv, no_constraint
        '''
        manifold = None
        if form == 'w':
            manifold = Sphere(n_attrs)
        elif form == 'M':
            manifold = Grassmann(n_attrs, n_components)
        elif form == 'wM':
            manifold_unitvec = Sphere(n_attrs)
            manifold_orth = Grassmann(n_attrs, n_components)
            manifold = Product([manifold_unitvec, manifold_orth])
        elif form == 'Mv':
            manifold_orth = Grassmann(n_attrs, n_components)
            manifold_unitvec = Sphere(n_components)
            manifold = Product([manifold_orth, manifold_unitvec])
        elif form == 'wMv':
            manifold_unitvec1 = Sphere(n_attrs)
            manifold_orth = Grassmann(n_attrs, n_components)
            manifold_unitvec2 = Sphere(n_components)
            manifold = Product(
                [manifold_unitvec1, manifold_orth, manifold_unitvec2])
        elif form == 'no_constraint':
            manifold = ModifiedOblique(n_attrs, n_components)

        return manifold

    def _to_wMv(self, form, answer, n_attrs, n_components=None):
        '''
        form: w, M, wM, Mv, wMv, no_constraint
        '''
        w, M, v = (None, None, None)
        if form == 'w':
            w = answer * np.sqrt(n_attrs)
            M = np.identity(n_attrs)
            v = np.ones(n_attrs)
        elif form == 'M':
            w = np.ones(n_attrs)
            M = answer
            v = np.ones(n_components)
        elif form == 'wM':
            w = answer[0] * np.sqrt(n_attrs)
            M = answer[1]
            v = np.ones(n_components)
        elif form == 'Mv':
            w = np.ones(n_attrs)
            M = answer[0]
            v = answer[1] * np.sqrt(n_components)
        elif form == 'wMv':
            w = answer[0] * np.sqrt(n_attrs)
            M = answer[1]
            v = answer[2] * np.sqrt(n_components)
        elif form == 'no_constraint':
            w = np.ones(n_attrs)
            M = answer
            v = np.ones(n_components)

        return w, M, v

    def _gen_cost_func_mat_decomp(self, manifold, P):

        @pymanopt.function.autograd(manifold)
        def _cost_func(w, M, v):
            P_ = np.diag(w) @ M @ np.diag(v)
            cost = np.sum(np.sqrt(np.sum((P - P_)**2, axis=1)))

            return cost

        return _cost_func

    def mat_decomp(self, P, form='wMv'):
        '''
        Parameters
        -------
        P: array-like, shape(n_attributes, n_latent_features)
            A projection matrix to be decomposed.
        form: string, optional (default: 'wMv')
            The form P will be decomposed to.
            Currently, only 'wMv' is supported. 
        Returns
        -------
        w: array-like, shape(n_attributes,)
            A vector for data scaling.
        M: array-like, shape(n_attributes, n_latent_features)
            A matrix for orthogonal projection.
        v: array-like, shape(n_latent_features,)
            A vector for scaling columns of M
        '''
        manifold_vec1 = ModifiedEuclidean((P.shape[0], ))
        manifold_vec2 = ModifiedEuclidean((P.shape[1], ))
        manifold_orth = Stiefel(P.shape[0], P.shape[1])
        manifold = Product([manifold_vec1, manifold_orth, manifold_vec2])

        cost_func = self._gen_cost_func_mat_decomp(manifold, P)

        problem = pymanopt.Problem(manifold=manifold, cost=cost_func)

        optimizer = SteepestDescent()
        optimizer._verbosity = 0

        w = None
        M = None
        v = None
        cost = np.inf
        for i in range(self.n_trials_for_mat_decomp):
            w_, M_, v_ = optimizer.run(problem).point
            cost_ = problem.cost([w_, M_, v_])
            if cost_ < cost:
                w = w_
                M = M_
                v = v_
                cost = cost_

        return w, M, v

    def fit(self,
            X,
            Gs,
            n_components=None,
            lasso_coeff=0,
            ridge_coeff=0,
            gd_params={},
            gd_preprocessed_data=None):
        """Perform feature learning on input data.

        Parameters
        ----------
        X: array-like, shape(n_samples, n_attributes)
            Target data.
        Gs: list of graphs, optional, (default=[])
            Already produced graphs that will be referred when producing a new
            graph (if existed). A set of graphs, Gi, in the paper.
        n_components: int, optional, (default=None)
            Number of components to take if generating linear transformations
            (e.g., "wMv" in the above paper). m' in the paper. This is not needed to
            be indicated if only applying feature scaling ("w" in the above paper).
        lasso_coeff: float, optional, (default=0)
            Coefficient for Lasso/L1-penalty. \lambda_1 in the paper.
        ridge_coeff: float, optional, (default=0)
            Coefficient for Ridge/L2-penalty. \lambda_2 in the paper.
        gd_params: parameters used for graph dissimlarity calculation.
        gd_preprocessed_data: preprocessed dissimilarities to speed up computations.
        Returns
        -------
        self.
        """
        n_attrs = X.shape[1]
        if n_components == None and not self.form == 'w':
            self.form = 'w'
            print('because n_components is None, w is used as form')

        manifold = self._gen_manifold(form=self.form,
                                      n_attrs=n_attrs,
                                      n_components=n_components)
        cost_func = self._gen_cost_func(
            manifold=manifold,
            form=self.form,
            X=X,
            Gs=Gs,
            lasso_coeff=lasso_coeff,
            ridge_coeff=ridge_coeff,
            gd_params=gd_params,
            gd_preprocessed_data=gd_preprocessed_data)
        self.problem = pymanopt.Problem(manifold=manifold, cost=cost_func)

        answer = self.optimizer.run(self.problem).point
        if isinstance(answer, tuple):
            # when returning non-best solutions as well
            best = answer[0]
            non_bests = answer[1]
            self.ws, self.Ms, self.vs, self.Ps = ([], [], [], [])
            for ans in non_bests:
                w, M, v = self._to_wMv(form=self.form,
                                       answer=ans,
                                       n_attrs=n_attrs,
                                       n_components=n_components)
                self.ws.append(w)
                self.Ms.append(M)
                self.vs.append(v)
                self.Ps.append(np.diag(w) @ M @ np.diag(v))
        else:
            # when returning only best answer
            best = answer

        self.w, self.M, self.v = self._to_wMv(form=self.form,
                                              answer=best,
                                              n_attrs=n_attrs,
                                              n_components=n_components)
        self.P = np.diag(self.w) @ self.M @ np.diag(self.v)

        return self
