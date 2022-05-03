#
# Implementation of Optimal Set of Projections by Lehmann and Theisel, 2015.
#

import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import TrustRegions
from pymanopt.solvers.steepest_descent import SteepestDescent


class OptSetProj():

    def __init__(self, solver=SteepestDescent()):
        self.solver = solver
        self.B = None

    def _gen_cost_func(self, manifold, X, As):

        @pymanopt.function.autograd(manifold)
        def _cost_func(B):
            # Note: in the equations in Optimal Sets of Projections, instances
            # correspond to cols and attrs correspond to rows.
            # This function follows their order during the calculation
            n, m = (X.T).shape
            r = len(As)

            Data_ = np.vstack(((X.T), np.ones(m)))

            stacked_As = None
            for A in As:
                if stacked_As is None:
                    stacked_As = A.T
                else:
                    stacked_As = np.vstack((stacked_As, A.T))
            A_ = np.vstack((stacked_As, np.zeros(n)))
            last_column_A_ = np.zeros((2 * r + 1, 1))
            last_column_A_[-1, -1] = 1
            A_ = np.hstack((A_, last_column_A_))

            B_ = np.vstack((B.T, np.zeros(n)))
            last_column_B_ = np.zeros((3, 1))
            last_column_B_[-1, -1] = 1
            B_ = np.hstack((B_, last_column_B_))

            D_ = Data_ @ Data_.T
            I_ = np.identity(n + 1)
            H_ = (I_ - D_ @ A_.T @ np.linalg.pinv(A_ @ D_ @ A_.T) @ A_) @ Data_

            E_ = B_ @ H_

            # dissim = np.trace(E_.T @ E_) / m
            dissim = np.sum((E_ * E_)) / m

            cost = 1e+3 / dissim

            return cost

        return _cost_func

    def fit(self, X, As):
        manifold = Grassmann(X.shape[1], 2)

        cost_func = self._gen_cost_func(manifold, X, As=As)

        problem = pymanopt.Problem(manifold=manifold, cost=cost_func)

        B = self.solver.solve(problem)
        self.B = B

        return self

    def starndard_config_projection(self, d):
        n = d  # to use the same symbol with the original paper
        alpha = 2 * np.pi / n
        b = np.sqrt(2 / n)
        A_pi = np.zeros((2, n))
        for i in range(n):
            A_pi[0, i] = b * np.sin(i * alpha)
            A_pi[1, i] = b * np.cos(i * alpha)

        return A_pi.T
