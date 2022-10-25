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

from pymanopt.optimizers.particle_swarm import ParticleSwarm
from pymanopt.optimizers.steepest_descent import SteepestDescent


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

    def __init__(self,
                 graph_func,
                 graph_dissim,
                 graph_dissim_reduce_func=np.min,
                 optimizer=ParticleSwarm(),
                 form='no_constraint',
                 n_trials_for_mat_decomp=5):
        '''
        form: {'w', 'M', 'wM', 'Mv', 'wMv', 'no_constraint'}
        '''
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
        entropy_sum = 0

        if lasso_coeff != 0:
            l1_sum = np.abs(P).sum(axis=0).sum()

        # here l2 reg is applied to compare each column of P
        if ridge_coeff != 0:
            l2_sum = np.sqrt((P**2).sum(axis=1)).sum()

        # here entropy reg is applied to compare each column of P
        if entropy_coeff != 0:
            # make every number positive
            P_ = (P - P.min(axis=0)) / (P.max(axis=0) -
                                        P.min(axis=0)) + np.finfo(float).eps
            entropy_sum = -(P_ * np.log(P_)).sum(axis=1).sum()

        # print(lasso_coeff * l1_sum, ridge_coeff * l2_sum,
        #       entropy_coeff * entropy_sum)

        return lasso_coeff * l1_sum + ridge_coeff * l2_sum + entropy_coeff * entropy_sum

    def _eval_cost(self,
                   X,
                   P,
                   Gs,
                   lasso_coeff=0,
                   ridge_coeff=0,
                   entropy_coeff=0,
                   gd_params={},
                   gd_preprocessed_data=None):
        new_G = self._construct_graph(X, P)

        dissim = self._eval_dissim_graph(
            new_G,
            Gs,
            gd_params=gd_params,
            gd_preprocessed_data=gd_preprocessed_data)
        regularization_penalty = self._regularization_penalty(
            P,
            lasso_coeff=lasso_coeff,
            ridge_coeff=ridge_coeff,
            entropy_coeff=entropy_coeff)

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
                       entropy_coeff=0,
                       gd_params={},
                       gd_preprocessed_data=None):
        '''
        form: w, M, wM, Mv, wMv, no_constraint
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
                    entropy_coeff=entropy_coeff,
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
                    entropy_coeff=entropy_coeff,
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
                    entropy_coeff=entropy_coeff,
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
                    entropy_coeff=entropy_coeff,
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
                    entropy_coeff=entropy_coeff,
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
                    entropy_coeff=entropy_coeff,
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
            entropy_coeff=0,
            multiple_answers=False,
            gd_params={},
            gd_preprocessed_data=None):
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
            entropy_coeff=entropy_coeff,
            gd_params=gd_params,
            gd_preprocessed_data=gd_preprocessed_data)
        self.problem = pymanopt.Problem(manifold=manifold, cost=cost_func)

        best, answers = (None, None)
        if multiple_answers:
            best, answers = self.optimizer.run(self.problem,
                                               multiple_answers=True).point
        else:
            best = self.optimizer.run(self.problem).point

        self.w, self.M, self.v = self._to_wMv(form=self.form,
                                              answer=best,
                                              n_attrs=n_attrs,
                                              n_components=n_components)
        self.P = np.diag(self.w) @ self.M @ np.diag(self.v)

        if not answers == None:
            self.ws, self.Ms, self.vs, self.Ps = ([], [], [], [])
            for ans in answers:
                w, M, v = self._to_wMv(form=self.form,
                                       answer=ans,
                                       n_attrs=n_attrs,
                                       n_components=n_components)
                self.ws.append(w)
                self.Ms.append(M)
                self.vs.append(v)
                self.Ps.append(np.diag(w) @ M @ np.diag(v))

        return self
