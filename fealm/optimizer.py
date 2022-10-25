#
# Enhanced versions of Pymanopt's NelderMead and ParticleSwarm
#
# NelderMead is enhanced by
# - adding random search for initial solutions,
# - employing the adaptive Nelder-Mead for the parameter selection,
# - applying multiprocessing using Pathos
#
# ParticleSwarm is enhanced by:
# - applying multiprocessing using Pathos
# - adding a functionality to return non-best solutions
#

# Pymanopt's implentation is under BSD-3-Clause license and Copyright (c) 2015-2016, Pymanopt Developers.

import time
import functools

import numpy as np
import numpy.random as rnd

from pymanopt.optimizers.optimizer import Optimizer

# faster but memory leak can happen
from pathos.multiprocessing import ProcessPool as Pool

import time

import numpy as np

import pymanopt
from pymanopt import tools
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.optimizers.steepest_descent import SteepestDescent

from pymanopt.optimizers.nelder_mead import compute_centroid

from func_timeout import func_set_timeout, FunctionTimedOut


class AdaptiveNelderMead(Optimizer):
    """Nelder-Mead alglorithm.

    Perform optimization using the derivative-free Nelder-Mead minimization
    algorithm.

    Args:
        max_cost_evaluations: Maximum number of allowed cost function
            evaluations.
        max_iterations: Maximum number of allowed iterations.
        reflection: Determines how far to reflect away from the worst vertex:
            stretched (reflection > 1), compressed (0 < reflection < 1),
            or exact (reflection = 1).
        expansion: Factor by which to expand the reflected simplex.
        contraction: Factor by which to contract the reflected simplex.
    """

    def __init__(
        self,
        max_cost_evaluations=None,
        max_iterations=None,
        reflection=None,
        expansion=None,
        contraction=None,
        shrink=None,
        randopt_population_size=None,
        n_jobs=-1,
        single_evaluation_max_time=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._max_cost_evaluations = max_cost_evaluations
        self._max_iterations = max_iterations
        self._reflection = reflection
        self._expansion = expansion
        self._contraction = contraction
        self._shrink = shrink
        self._n_jobs = n_jobs
        self._randopt_population_size = randopt_population_size
        self.single_evaluation_max_time = single_evaluation_max_time

    def compute_euclid_centroid(self, manifold, points):
        return functools.reduce(lambda a, b: a + b, points) / len(points)

    def run(self, problem, *, initial_point=None) -> OptimizerResult:
        manifold = problem.manifold
        if 'Oblique' in manifold._name or 'Euclidean' in manifold._name:
            compute_centroid_ = self.compute_euclid_centroid
        else:
            compute_centroid_ = compute_centroid

        if self.single_evaluation_max_time is None:
            self.single_evaluation_max_time = self.max_time

        @func_set_timeout(self.single_evaluation_max_time)
        def cost(x):
            return problem.cost(x)

        def objective(x):
            try:
                val = cost(x)
            except FunctionTimedOut:
                print('cost evaluation timeout')
                val = np.finfo(float).max
            return val

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = manifold.dim
        if self._max_cost_evaluations is None:
            self._max_cost_evaluations = max(1000, 2 * dim)
        if self._max_iterations is None:
            self._max_iterations = max(2000, 4 * dim)

        # change parameter adaptively based on dim
        if dim < 2:
            self._reflection = 1 if self._reflection is None else self._reflection
            self._expansion = 2 if self._expansion is None else self._expansion
            self._contraction = 0.5 if self._contraction is None else self._contraction
            self._shrink = 0.5 if self._shrink is None else self._shrink
        if dim >= 2:
            self._reflection = 1 if self._reflection is None else self._reflection
            self._expansion = 1 + 2 / dim if self._expansion is None else self._expansion
            self._contraction = 0.75 - 0.5 / dim if self._contraction is None else self._contraction
            self._shrink = 1 - 1 / dim if self._shrink is None else self._shrink

        # If no initial simplex x is given by the user, generate one at random.
        ## Combining Random Search
        if self._randopt_population_size is None:
            self._randopt_population_size = int(dim * 10 + 1)
        # if self._randopt_population_size < dim + 1:
        #     self._randopt_population_size = int(dim + 1)

        if initial_point is None:
            x = [
                manifold.random_point()
                for _ in range(self._randopt_population_size)
            ]
        elif (tools.is_sequence(initial_point)
              and len(initial_point) != self._randopt_population_size):
            x = initial_point
        else:
            raise ValueError(
                "The initial simplex `initial_point` must be a sequence of "
                f"{self._randopt_population_size} points")

        # Compute objective-related quantities for x, and setup a function
        # evaluations counter.

        # costs = np.array([objective(xi) for xi in x])
        if self._n_jobs == 1:
            costs = [objective(xi) for xi in x]
        elif self._n_jobs > 1:
            with Pool(nodes=self._n_jobs) as p:
                costs = p.map(objective, x)
        else:
            with Pool() as p:
                costs = p.map(objective, x)
        costs = np.array(costs)
        cost_evaluations = len(x)

        # Sort simplex points by cost.
        order = np.argsort(costs)
        costs = costs[order]
        x = [x[i] for i in order]

        ## Keep only top n+1
        costs = costs[:dim + 1]
        x = x[:dim + 1]

        # Iteration counter (at any point, iteration is the number of fully
        # executed iterations so far).
        iteration = 0

        start_time = time.time()

        self._initialize_log()

        while True:
            iteration += 1

            # Sort simplex points by cost.
            order = np.argsort(costs)
            costs = costs[order]
            x = [x[i] for i in order]

            if self._verbosity >= 2:
                print(f"Cost evals: {cost_evaluations:7d}\t"
                      f"Best cost: {costs[0]:+.8e}")

            # Compute a centroid for the dim best points.
            xbar = compute_centroid_(manifold, x[:-1])

            # Compute the direction for moving along the axis xbar - worst x.
            vec = manifold.log(xbar, x[-1])
            pseudo_gradient_norm = np.linalg.norm(vec)

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                iteration=iteration,
                cost_evaluations=cost_evaluations,
                gradient_norm=pseudo_gradient_norm)
            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            # Reflection step
            # xr = manifold.retraction(xbar, -self._reflection * vec)
            xr = manifold.retraction((1 + self._reflection) * xbar,
                                     -self._reflection * vec)
            costr = objective(xr)
            cost_evaluations += 1

            # If the reflected point is honorable, drop the worst point,
            # replace it by the reflected point and start a new iteration.
            if costr >= costs[0] and costr < costs[-2]:
                if self._verbosity >= 2:
                    print("Reflection")
                costs[-1] = costr
                x[-1] = xr
                continue

            # If the reflected point is better than the best point, expand.
            if costr < costs[0]:
                # xe = manifold.retraction(xbar, -self._expansion * vec)
                xe = manifold.retraction((1 - self._expansion) * xbar,
                                         self._expansion * xr)

                coste = objective(xe)
                cost_evaluations += 1
                if coste < costr:
                    if self._verbosity >= 2:
                        print("Expansion")
                    costs[-1] = coste
                    x[-1] = xe
                    continue
                else:
                    if self._verbosity >= 2:
                        print("Reflection (failed expansion)")
                    costs[-1] = costr
                    x[-1] = xr
                    continue

            # If the reflected point is worse than the second to worst point,
            # contract.
            if costr >= costs[-2]:
                if costr < costs[-1]:
                    # do an outside contraction
                    # xoc = manifold.retraction(xbar, -self._contraction * vec)
                    xoc = manifold.retraction((1 - self._contraction) * xbar,
                                              self._contraction * xr)

                    costoc = objective(xoc)
                    cost_evaluations += 1
                    if costoc <= costr:
                        if self._verbosity >= 2:
                            print("Outside contraction")
                        costs[-1] = costoc
                        x[-1] = xoc
                        continue
                else:
                    # do an inside contraction

                    # xic = manifold.retraction(xbar, self._contraction * vec)
                    xic = manifold.retraction((1 + self._contraction) * xbar,
                                              -self._contraction * vec)
                    costic = objective(xic)
                    cost_evaluations += 1
                    if costic <= costs[-1]:
                        if self._verbosity >= 2:
                            print("Inside contraction")
                        costs[-1] = costic
                        x[-1] = xic
                        continue

            # If we get here, shrink the simplex around x[0].
            if self._verbosity >= 2:
                print("Shrinkage")
            x0 = x[0]
            for i in np.arange(1, dim + 1):
                # x[i] = manifold.pair_mean(x0, x[i])
                x[i] = manifold.retraction((1 - self._shrink) * x0,
                                           self._shrink * x[i])
                # costs[i] = objective(x[i])

            if self._n_jobs == 1:
                costs = [objective(xi) for xi in x]
            elif self._n_jobs > 1:
                with Pool(nodes=self._n_jobs) as p:
                    costs = p.map(objective, x)
            else:
                with Pool() as p:
                    costs = p.map(objective, x)
            costs = np.array(costs)
            cost_evaluations += dim

        x = x[0]
        cost = objective(x)
        return self._return_result(
            start_time=start_time,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=cost_evaluations,
        )


class ParticleSwarm(Optimizer):
    """Particle swarm optimization (PSO) method.
    Perform optimization using the derivative-free particle swarm optimization
    algorithm.
    Args:
        max_cost_evaluations: Maximum number of allowed cost evaluations.
        max_iterations: Maximum number of allowed iterations.
        population_size: Size of the considered swarm population.
        nostalgia: Quantifies performance relative to past performances.
        social: Quantifies performance relative to neighbors.
    """

    def __init__(
        self,
        max_cost_evaluations=None,
        max_iterations=None,
        population_size=None,
        nostalgia=1.4,
        social=1.4,
        n_jobs=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._max_cost_evaluations = max_cost_evaluations
        self._max_iterations = max_iterations
        self._population_size = population_size
        self._nostalgia = nostalgia
        self._social = social
        self._n_jobs = n_jobs

    def run(self, problem, x=None, multiple_answers=False):
        """Run PSO algorithm.
        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
            x: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        man = problem.manifold
        verbosity = self._verbosity
        objective = problem.cost

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = man.dim
        if self._max_cost_evaluations is None:
            self._max_cost_evaluations = max(5000, 2 * dim)
        if self._max_iterations is None:
            self._max_iterations = max(500, 4 * dim)
        if self._population_size is None:
            self._population_size = min(40, 10 * dim)

        # If no initial population x is given by the user, generate one at
        # random.
        if x is None:
            x = [man.random_point() for i in range(int(self._population_size))]
        elif not hasattr(x, "__iter__"):
            raise ValueError("The initial population x must be iterable")
        else:
            if len(x) != self._population_size:
                print("The population size was forced to the size of "
                      "the given initial population")
                self._population_size = len(x)

        # Initialize personal best positions to the initial population.
        y = list(x)

        # Save a copy of the swarm at the previous iteration.
        xprev = list(x)

        if self._n_jobs > 0:
            with Pool(nodes=self._n_jobs) as p:
                # Initialize velocities for each particle.
                v = [man.random_tangent_vector(xi) for xi in x]
                # Compute cost for each particle xi.
                costs = p.map(objective, x)
        else:
            with Pool() as p:
                # Initialize velocities for each particle.
                v = [man.random_tangent_vector(xi) for xi in x]
                # Compute cost for each particle xi.
                costs = p.map(objective, x)

        fy = list(costs)
        cost_evaluations = self._population_size

        # Identify the best particle and store its cost/position.
        imin = np.array(costs).argmin()
        fbest = costs[imin]
        xbest = x[imin]

        self._initialize_log()

        # Iteration counter (at any point, iter is the number of fully executed
        # iterations so far).
        iteration = 0
        start_time = time.time()

        while True:
            iteration += 1

            print(
                f'iteration: {iteration}, cost: {cost_evaluations}, best: {fbest}'
            )

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                iteration=iteration,
                cost_evaluations=cost_evaluations)

            if stopping_criterion:
                if verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            # Compute the inertia factor which we linearly decrease from 0.9 to
            # 0.4 from iteration = 0 to iteration = max_iterations.
            w = 0.4 + 0.5 * (1 - iteration / self._max_iterations)

            # Compute the velocities.
            for i, xi in enumerate(x):
                # Get the position and past best position of particle i.
                yi = y[i]

                # Get the previous position and velocity of particle i.
                xiprev = xprev[i]
                vi = v[i]

                # Compute the new velocity of particle i, composed of three
                # contributions.
                inertia = w * man.transport(xiprev, xi, vi)
                nostalgia = rnd.rand() * self._nostalgia * man.log(xi, yi)
                social = rnd.rand() * self._social * man.log(xi, xbest)

                v[i] = inertia + nostalgia + social

            # Backup the current swarm positions.
            xprev = list(x)

            if self._n_jobs > 0:
                with Pool(nodes=self._n_jobs) as p:
                    costs = p.map(objective, x)
            else:
                with Pool() as p:
                    costs = p.map(objective, x)

            # update the bests
            for i, fxi in enumerate(costs):
                # Update self-best if necessary.
                if fxi < fy[i]:
                    fy[i] = fxi
                    y[i] = x[i]
                    # Update global best if necessary.
                    if fy[i] < fbest:
                        fbest = fy[i]
                        xbest = x[i]

            # Compute new position of particle i.
            x = [man.retraction(xi, vi) for xi, vi in zip(x, v)]

            cost_evaluations += self._population_size

        if multiple_answers:
            point = (xbest, x)
        else:
            point = xbest

        return self._return_result(
            start_time=start_time,
            point=point,
            cost=fbest,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=cost_evaluations,
        )


class RandomAnswer(Optimizer):

    def __init__(
        self,
        populationsize=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._population_size = populationsize

    def run(self, problem, x=None):
        """Return random params on a specified manifold
        """
        start_time = time.time()
        man = problem.manifold
        point = man.random_point()

        return self._return_result(
            start_time=start_time,
            point=point,
        )
